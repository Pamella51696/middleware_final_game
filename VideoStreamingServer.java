import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class VideoStreamingServer {

    private static final int DEFAULT_PORT  = 9090;
    private static final int TARGET_HEIGHT = 360;
    private static final int TARGET_WIDTH  = 640;
    private static final int OVERLAP_PX    = 80;
    /** Last stitch panel — rear camera (bumper at bottom of raw fisheye). */
    private static final int REAR_CAMERA_INDEX = 3;

    // =========================================================================
    public static void main(String[] args) throws IOException {

        Path frontVideo = ensureDecodable(Paths.get("rear_1.mov"));
        Path rearVideo  = ensureDecodable(Paths.get("left_1.mov"));
        Path sideVideo  = ensureDecodable(Paths.get("right_1.mov"));
        Path backVideo  = ensureDecodable(Paths.get("front_1.mov"));

        if (args.length >= 5) {
            frontVideo = ensureDecodable(Paths.get(args[1]));
            rearVideo  = ensureDecodable(Paths.get(args[2]));
            sideVideo  = ensureDecodable(Paths.get(args[3]));
            backVideo  = ensureDecodable(Paths.get(args[4]));
        }

        Path[] videos = { frontVideo, rearVideo, sideVideo, backVideo };

        for (Path v : videos) {
            if (!Files.exists(v) || Files.isDirectory(v)) {
                System.err.println("Video file not found: " + v);
                return;
            }
        }

        int port = args.length > 0 ? Integer.parseInt(args[0]) : DEFAULT_PORT;

        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("OpenCV native library not found: " + e.getMessage());
            return;
        }
        loadFfmpegPlugin();

        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);

        server.createContext("/stitch", new StitchHandler(videos));
        server.createContext("/play",   new PlayerPageHandler());

        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();

        System.out.println("Server started  ->  http://localhost:" + port + "/play");
        System.out.println("Feeds: front=" + frontVideo + " rear=" + rearVideo
                + " side=" + sideVideo + " back=" + backVideo);
    }

    // STITCH HANDLER  -  undistort each feed, then feather-blend panorama

    private static class StitchHandler implements HttpHandler {
      private final Path[] videoFiles;
      StitchHandler(Path[] f) { this.videoFiles = f; }

      @Override public void handle(HttpExchange ex) throws IOException {
        if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
          ex.sendResponseHeaders(405, -1); return;
        }

        VideoCapture[] caps = new VideoCapture[videoFiles.length];
        for (int i = 0; i < videoFiles.length; i++) {
          caps[i] = openVideo(videoFiles[i]);
          if (caps[i] == null || !caps[i].isOpened()) {
            System.err.println("Could not open video: " + videoFiles[i]);
            ex.sendResponseHeaders(500, -1); return;
          }
        }

        CameraFeedFilter[] undistort = new CameraFeedFilter[videoFiles.length];
        for (int i = 0; i < undistort.length; i++) {
          undistort[i] = (i == REAR_CAMERA_INDEX)
              ? new RearFeedPipeline()
              : new FisheyeUndistorter();
        }

        ex.getResponseHeaders().set("Content-Type", "multipart/x-mixed-replace; boundary=frame");
        ex.sendResponseHeaders(200, 0);

        try (OutputStream out = ex.getResponseBody()) {
          Mat[] frames  = new Mat[videoFiles.length];
          Mat[] ready   = new Mat[videoFiles.length];
          for (int i = 0; i < videoFiles.length; i++) {
            frames[i] = new Mat();
            ready[i]  = new Mat();
          }

          while (true) {
            for (int i = 0; i < caps.length; i++) {
              if (!readOrLoop(caps, i, videoFiles[i], frames[i])) {
                continue;
              }
              undistort[i].recalibrateAndFilter(frames[i], ready[i]);
            }

            boolean allReady = true;
            for (int i = 0; i < ready.length; i++) {
              if (ready[i].empty()
                  || ready[i].cols() != TARGET_WIDTH
                  || ready[i].rows() != TARGET_HEIGHT) {
                allReady = false;
                break;
              }
            }
            if (!allReady) {
              continue;
            }

            Mat panorama = featherStitch(ready);
            writeFrame(out, encodeJpeg(panorama));
            panorama.release();
          }
        }
        finally {
          for (VideoCapture c : caps) c.release();
        }

      } // handle()
    } // StitchHandler


    // PLAYER PAGE  -  stitched panorama only, full-viewport

    private static class PlayerPageHandler implements HttpHandler {

        @Override public void handle(HttpExchange ex) throws IOException {
            String html = "<!DOCTYPE html><html lang='en'><head>"
                + "<meta charset='UTF-8'>"
                + "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                + "<title>360° Panoramic View</title>"
                + "<style>"
                + "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }"
                + "html, body { height: 100%; background: #0a0a0f; color: #e0e0e0;"
                + "  font-family: 'Segoe UI', sans-serif; overflow: hidden; }"
                + ".container { display: flex; flex-direction: column;"
                + "  align-items: center; justify-content: center;"
                + "  height: 100vh; padding: 16px; gap: 12px; }"
                + "h1 { font-size: 1.4rem; font-weight: 300; letter-spacing: 2px;"
                + "  color: #7ec8e3; text-align: center; flex-shrink: 0; }"
                + ".pano-wrap { width: 100%; flex: 1; min-height: 0;"
                + "  border: 1px solid #2a2a3a; border-radius: 8px; overflow: hidden;"
                + "  display: flex; align-items: center; justify-content: center; }"
                + ".pano-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; }"
                + "</style>"
                + "</head><body>"
                + "<div class='container'>"
                + "  <h1>360° Panoramic Camera System</h1>"
                + "  <div class='pano-wrap'>"
                + "    <img src='/stitch' alt='360° stitched panorama'>"
                + "  </div>"
                + "</div>"
                + "</body></html>";

            byte[] bytes = html.getBytes("UTF-8");
            ex.getResponseHeaders().set("Content-Type", "text/html; charset=UTF-8");
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
        }
    }


    interface CameraFeedFilter {
        void recalibrateAndFilter(Mat src, Mat dst360x640);
    }

    //  FISHEYE UNDISTORT LAYER
    //
    //  Size-adaptive: K is rebuilt from frame size. Do not send 180° fisheye
    //  through a pinhole model — tan(90°) is infinite and produces the
    //  radial starburst. Output is a finite rectilinear crop (default 90°).

    static final class FisheyeUndistorter implements CameraFeedFilter {

        /** Assumed diagonal-ish horizontal coverage of the raw fisheye. Keep < 170. */
        private static final double INPUT_FOV_DEG = 150.0;

        /** Rectilinear view sent to stitch. 80–100 is typical; lower = more zoom. */
        private static final double OUTPUT_FOV_DEG = 90.0;

        private static final double[] FISHEYE_D = { 0.0, 0.0, 0.0, 0.0 };

        private Mat map1;
        private Mat map2;
        private Mat undistorted;
        private Mat filtered;
        private int cachedSrcW = -1;
        private int cachedSrcH = -1;

        @Override
        public void recalibrateAndFilter(Mat src, Mat dst360x640) {
            if (src == null || src.empty()) {
                return;
            }

            ensureMaps(src.cols(), src.rows());

            if (undistorted == null) undistorted = new Mat();
            if (filtered == null)    filtered    = new Mat();

            Imgproc.remap(src, undistorted, map1, map2, Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT);

            Imgproc.GaussianBlur(undistorted, filtered, new Size(3, 3), 0.6);

            Size target = new Size(TARGET_WIDTH, TARGET_HEIGHT);
            if (filtered.cols() == TARGET_WIDTH && filtered.rows() == TARGET_HEIGHT) {
                filtered.copyTo(dst360x640);
            } else {
                Imgproc.resize(filtered, dst360x640, target, 0, 0, Imgproc.INTER_AREA);
            }
        }

        private void ensureMaps(int srcW, int srcH) {
            if (map1 != null && srcW == cachedSrcW && srcH == cachedSrcH) {
                return;
            }

            Size dstSize = new Size(TARGET_WIDTH, TARGET_HEIGHT);

            Mat K = equidistantK(srcW, srcH, INPUT_FOV_DEG);
            Mat D = distortionCoeffs();
            Mat R = Mat.eye(3, 3, CvType.CV_64FC1);
            Mat P = pinholeK(TARGET_WIDTH, TARGET_HEIGHT, OUTPUT_FOV_DEG);

            if (map1 == null) map1 = new Mat();
            if (map2 == null) map2 = new Mat();

            Calib3d.fisheye_initUndistortRectifyMap(
                    K, D, R, P, dstSize, CvType.CV_16SC2, map1, map2);

            cachedSrcW = srcW;
            cachedSrcH = srcH;

            K.release();
            D.release();
            R.release();
            P.release();
        }

        /** r = f * theta. fx == fy so aspect ratio is preserved. */
        static Mat equidistantK(int width, int height, double fovDeg) {
            double half = Math.toRadians(fovDeg) / 2.0;
            double f = (Math.max(width, height) / 2.0) / half;
            return matrixK(f, f, width / 2.0, height / 2.0);
        }

        /** r = f * tan(theta). FOV must stay well below 180°. */
        static Mat pinholeK(int width, int height, double fovDeg) {
            double half = Math.toRadians(fovDeg) / 2.0;
            double f = (width / 2.0) / Math.tan(half);
            return matrixK(f, f, width / 2.0, height / 2.0);
        }

        static Mat matrixK(double fx, double fy, double cx, double cy) {
            Mat K = Mat.eye(3, 3, CvType.CV_64FC1);
            K.put(0, 0, fx);
            K.put(1, 1, fy);
            K.put(0, 2, cx);
            K.put(1, 2, cy);
            return K;
        }

        static Mat distortionCoeffs() {
            Mat D = new Mat(4, 1, CvType.CV_64FC1);
            D.put(0, 0, FISHEYE_D[0]);
            D.put(1, 0, FISHEYE_D[1]);
            D.put(2, 0, FISHEYE_D[2]);
            D.put(3, 0, FISHEYE_D[3]);
            return D;
        }

        static Mat eulerRyxz(double pitchDeg, double yawDeg, double rollDeg) {
            Mat rvec = new Mat(3, 1, CvType.CV_64FC1);
            rvec.put(0, 0, Math.toRadians(pitchDeg));
            rvec.put(1, 0, Math.toRadians(yawDeg));
            rvec.put(2, 0, Math.toRadians(rollDeg));
            Mat R = new Mat();
            Calib3d.Rodrigues(rvec, R);
            rvec.release();
            return R;
        }
    }


    //  REAR CAMERA PIPELINE
    //
    //  The rear fisheye sits low and looks down: bumper / plate fill the bottom
    //  of the circle and the street sits in the upper FOV. A separate pitch-up
    //  remap + top crop keeps horizon/road and drops the number plate.
    //  Only this block is used for stitch index REAR_CAMERA_INDEX.

    static final class RearFeedPipeline implements CameraFeedFilter {

        /** Degrees to tilt the virtual camera toward the top of the raw frame. */
        private static final double LOOK_UP_DEG = 18.0;

        /** Clockwise-positive in the image. Set negative to counter a clockwise roll. */
        private static final double ROLL_DEG = 0.0;

        private static final double INPUT_FOV_DEG  = 160.0;
        private static final double OUTPUT_FOV_DEG = 88.0;

        /** Skip this much from the top of the remapped frame (camera housing). */
        private static final double TOP_SKIP_FRACTION = 0.10;

        /**
         * Vertical window after skip. Higher includes more road / bumper;
         * lower stays on trees/sky.
         */
        private static final double KEEP_FRACTION = 0.72;

        private Mat map1;
        private Mat map2;
        private Mat undistorted;
        private int cachedSrcW = -1;
        private int cachedSrcH = -1;
        private int mapH = TARGET_HEIGHT;

        @Override
        public void recalibrateAndFilter(Mat src, Mat dst360x640) {
            if (src == null || src.empty()) {
                return;
            }

            ensureMaps(src.cols(), src.rows());

            if (undistorted == null) undistorted = new Mat();

            Imgproc.remap(src, undistorted, map1, map2, Imgproc.INTER_LINEAR);

            int h = undistorted.rows();
            int skip = (int) Math.round(h * TOP_SKIP_FRACTION);
            int keepH = Math.max(1, (int) Math.round(h * KEEP_FRACTION));
            if (skip + keepH > h) {
                keepH = h - skip;
            }
            if (keepH < 1) {
                skip = 0;
                keepH = h;
            }
            Mat window = undistorted.rowRange(skip, skip + keepH);
            Imgproc.resize(window, dst360x640, new Size(TARGET_WIDTH, TARGET_HEIGHT),
                    0, 0, Imgproc.INTER_AREA);
        }

        private void ensureMaps(int srcW, int srcH) {
            if (map1 != null && srcW == cachedSrcW && srcH == cachedSrcH) {
                return;
            }

            mapH = (int) Math.round(TARGET_HEIGHT / KEEP_FRACTION);
            Size dstSize = new Size(TARGET_WIDTH, mapH);

            Mat K = FisheyeUndistorter.equidistantK(srcW, srcH, INPUT_FOV_DEG);
            Mat D = FisheyeUndistorter.distortionCoeffs();
            // Negative pitch (Y-down camera) aims the virtual view at the top of the fisheye.
            Mat R = FisheyeUndistorter.eulerRyxz(-LOOK_UP_DEG, 0.0, ROLL_DEG);
            Mat P = FisheyeUndistorter.pinholeK(TARGET_WIDTH, mapH, OUTPUT_FOV_DEG);

            if (map1 == null) map1 = new Mat();
            if (map2 == null) map2 = new Mat();

            Calib3d.fisheye_initUndistortRectifyMap(
                    K, D, R, P, dstSize, CvType.CV_16SC2, map1, map2);

            cachedSrcW = srcW;
            cachedSrcH = srcH;

            K.release();
            D.release();
            R.release();
            P.release();
        }
    }


    //  CORE BLENDING  —  featherStitch

    static Mat featherStitch(Mat[] frames) {
        int N = frames.length;
        int H = TARGET_HEIGHT;
        int W = TARGET_WIDTH;

        int overlap = Math.min(OVERLAP_PX, W / 4);
        int panoW   = W + (N - 1) * (W - overlap);

        Mat accumColor  = Mat.zeros(H, panoW, CvType.CV_32FC3);
        Mat accumWeight = Mat.zeros(H, panoW, CvType.CV_32FC1);

        for (int i = 0; i < N; i++) {
            int xStart = i * (W - overlap);

            Mat weight = buildFeatherMask(H, W, overlap);

            Mat frameF = new Mat();
            frames[i].convertTo(frameF, CvType.CV_32FC3);

            Mat weight3 = new Mat();
            List<Mat> ch = new ArrayList<>();
            ch.add(weight); ch.add(weight); ch.add(weight);
            Core.merge(ch, weight3);

            Mat wFrame = new Mat();
            Core.multiply(frameF, weight3, wFrame);

            int xEnd    = Math.min(xStart + W, panoW);
            int wActual = xEnd - xStart;

            Mat colorRoi  = accumColor.submat(0, H, xStart, xEnd);
            Mat weightRoi = accumWeight.submat(0, H, xStart, xEnd);

            Mat wFrameCrop = wFrame.colRange(0, wActual);
            Mat weightCrop = weight.colRange(0, wActual);

            Core.add(colorRoi,  wFrameCrop, colorRoi);
            Core.add(weightRoi, weightCrop, weightRoi);

            colorRoi.release(); weightRoi.release();
            frameF.release(); weight.release(); weight3.release();
            wFrame.release();
        }

        Mat safeW = new Mat();
        Core.max(accumWeight, new Scalar(1e-6), safeW);

        Mat safeW3 = new Mat();
        List<Mat> wch = new ArrayList<>();
        wch.add(safeW); wch.add(safeW); wch.add(safeW);
        Core.merge(wch, safeW3);

        Mat blended = new Mat();
        Core.divide(accumColor, safeW3, blended);

        Mat result = new Mat();
        blended.convertTo(result, CvType.CV_8UC3);

        accumColor.release(); accumWeight.release();
        safeW.release(); safeW3.release(); blended.release();

        return result;
    }

    static Mat buildFeatherMask(int H, int W, int overlap) {
        Mat mask = new Mat(H, W, CvType.CV_32FC1, new Scalar(1.0));
        for (int x = 0; x < overlap; x++) {
            float alpha = (float) x / overlap;
            for (int y = 0; y < H; y++) {
                mask.put(y, x,          new float[]{ alpha });
                mask.put(y, W - 1 - x,  new float[]{ alpha });
            }
        }
        return mask;
    }


    // VIDEO IO  —  FFmpeg plugin, then MSMF without RGB32 conversion

    static Path preferH264(Path requested) {
        return ensureDecodable(requested);
    }

    /**
     * These camera .mov files are PNG video (FFmpeg codec_id=61, fourcc png).
     * OpenCV's bundled FFmpeg and Windows MSMF cannot decode that.
     * Use a sibling H.264 .mp4, transcoding with ffmpeg when needed.
     */
    static Path ensureDecodable(Path requested) {
        Path mp4 = siblingWithExt(requested, ".mp4");
        if (isUsableFile(mp4)) {
            System.out.println("Using " + mp4.getFileName() + " (H.264) instead of "
                    + requested.getFileName());
            return mp4;
        }
        if (!isUsableFile(requested)) {
            return requested;
        }
        if (transcodeToH264(requested, mp4) && isUsableFile(mp4)) {
            return mp4;
        }
        System.err.println("Cannot decode " + requested.getFileName()
                + " (PNG-in-MOV). Install ffmpeg on PATH and re-run, or convert:");
        System.err.println("  ffmpeg -y -i " + requested.getFileName()
                + " -c:v libx264 -pix_fmt yuv420p -an " + mp4.getFileName());
        return requested;
    }

    static Path siblingWithExt(Path requested, String ext) {
        String name = requested.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String stem = dot >= 0 ? name.substring(0, dot) : name;
        Path dir = requested.toAbsolutePath().getParent();
        if (dir == null) {
            dir = Paths.get(".");
        }
        return dir.resolve(stem + ext);
    }

    static boolean isUsableFile(Path path) {
        try {
            return path != null && Files.isRegularFile(path) && Files.size(path) > 0;
        } catch (IOException e) {
            return false;
        }
    }

    static boolean transcodeToH264(Path src, Path dst) {
        System.out.println("Transcoding " + src.getFileName() + " -> " + dst.getFileName()
                + " (PNG MOV cannot be decoded by OpenCV FFmpeg/MSMF)");
        ProcessBuilder pb = new ProcessBuilder(
                "ffmpeg", "-hide_banner", "-y",
                "-i", src.toAbsolutePath().toString(),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-an", dst.toAbsolutePath().toString());
        pb.inheritIO();
        try {
            int code = pb.start().waitFor();
            if (code == 0 && isUsableFile(dst)) {
                System.out.println("Transcode OK: " + dst.getFileName());
                return true;
            }
            System.err.println("ffmpeg exited with code " + code);
        } catch (IOException e) {
            System.err.println("ffmpeg not found on PATH: " + e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("ffmpeg transcode interrupted");
        }
        try {
            Files.deleteIfExists(dst);
        } catch (IOException ignored) {
        }
        return false;
    }

    /**
     * Official Windows OpenCV puts opencv_java*.dll in build/java/x64 and the
     * FFmpeg videoio plugin in build/bin. java.library.path usually only has
     * the first folder, so MSMF is used and .mov RGB32 decode fails.
     */
    static void loadFfmpegPlugin() {
        String[] dllNames = {
            "opencv_videoio_ffmpeg490_64.dll",
            "opencv_videoio_ffmpeg4.dll",
            "opencv_videoio_ffmpeg.dll"
        };
        List<Path> dirs = new ArrayList<>();
        String libPath = System.getProperty("java.library.path", "");
        for (String dir : libPath.split(File.pathSeparator)) {
            if (dir == null || dir.trim().isEmpty()) continue;
            Path p = Paths.get(dir.trim()).toAbsolutePath().normalize();
            dirs.add(p);
            if (p.getParent() != null) {
                dirs.add(p.getParent());
                if (p.getParent().getParent() != null) {
                    dirs.add(p.getParent().getParent().resolve("bin"));
                }
            }
        }
        dirs.add(Paths.get("opencv", "build", "bin").toAbsolutePath());
        dirs.add(Paths.get("..", "opencv", "build", "bin").toAbsolutePath());

        for (Path dir : dirs) {
            for (String dll : dllNames) {
                Path candidate = dir.resolve(dll).normalize();
                if (!Files.isRegularFile(candidate)) continue;
                try {
                    System.load(candidate.toString());
                    System.out.println("Loaded FFmpeg videoio plugin: " + candidate);
                    return;
                } catch (Throwable t) {
                    System.err.println("Could not load " + candidate + ": " + t.getMessage());
                }
            }
        }
        System.err.println("FFmpeg videoio plugin not loaded. MSMF will be used for .mov "
                + "and may fail RGB32. Copy opencv_videoio_ffmpeg490_64.dll next to "
                + "opencv_java490.dll or into opencv\\build\\bin on PATH.");
    }

    static VideoCapture openVideo(Path path) {
        String file = path.toAbsolutePath().toString();

        int[] apis = {
            Videoio.CAP_FFMPEG,
            Videoio.CAP_ANY,
            Videoio.CAP_MSMF
        };
        String[] labels = { "FFMPEG", "ANY", "MSMF" };

        for (int a = 0; a < apis.length; a++) {
            for (double convert : new double[]{ (apis[a] == Videoio.CAP_MSMF ? 0 : 1), 0, 1 }) {
                VideoCapture cap = tryOpen(file, apis[a], labels[a], convert);
                if (cap != null) {
                    return cap;
                }
            }
        }

        VideoCapture cap = new VideoCapture();
        cap.open(file);
        if (cap.isOpened()) {
            cap.set(Videoio.CAP_PROP_CONVERT_RGB, 0);
            if (probeFrame(cap)) {
                logBackend(file, cap, "default");
                return cap;
            }
        }
        cap.release();
        return null;
    }

    static VideoCapture tryOpen(String file, int api, String label, double convertRgb) {
        VideoCapture cap = new VideoCapture();
        MatOfInt params = new MatOfInt(Videoio.CAP_PROP_CONVERT_RGB, (int) convertRgb);
        try {
            boolean opened = cap.open(file, api, params);
            if (!opened || !cap.isOpened()) {
                cap.release();
                params.release();
                return null;
            }
        } catch (Exception e) {
            cap.release();
            params.release();
            return null;
        }
        params.release();

        cap.set(Videoio.CAP_PROP_CONVERT_RGB, convertRgb);
        if (!probeFrame(cap)) {
            cap.release();
            return null;
        }
        logBackend(file, cap, label + " convertRGB=" + (int) convertRgb);
        return cap;
    }

    static boolean probeFrame(VideoCapture cap) {
        Mat probe = new Mat();
        boolean ok = cap.read(probe) && !probe.empty();
        if (ok) {
            Mat bgr = new Mat();
            ok = toBgr(probe, bgr);
            bgr.release();
        }
        probe.release();
        if (!ok) {
            return false;
        }
        cap.set(Videoio.CAP_PROP_POS_FRAMES, 0);
        cap.set(Videoio.CAP_PROP_POS_MSEC, 0);
        return true;
    }

    static void logBackend(String file, VideoCapture cap, String requested) {
        String backend = requested;
        try {
            String named = cap.getBackendName();
            if (named != null && !named.isEmpty()) {
                backend = named + " / " + requested;
            }
        } catch (Exception ignored) {
        }
        System.out.println("Opened " + file + " [" + backend + "]");
    }

    static boolean toBgr(Mat src, Mat dst) {
        if (src == null || src.empty()) {
            return false;
        }
        int type = src.type();
        if (type == CvType.CV_8UC3) {
            src.copyTo(dst);
            return true;
        }
        if (type == CvType.CV_8UC4) {
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGRA2BGR);
            return !dst.empty();
        }
        if (type == CvType.CV_8UC2) {
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_YUV2BGR_YUY2);
            return !dst.empty();
        }
        if (src.channels() == 1) {
            int h = src.rows();
            int w = src.cols();
            if (h * 2 % 3 == 0) {
                int visH = h * 2 / 3;
                if (visH > 0 && visH * 3 / 2 == h) {
                    try {
                        Imgproc.cvtColor(src, dst, Imgproc.COLOR_YUV2BGR_NV12);
                        if (!dst.empty() && dst.channels() == 3) {
                            return true;
                        }
                    } catch (Exception ignored) {
                    }
                }
            }
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_GRAY2BGR);
            return !dst.empty();
        }
        src.copyTo(dst);
        return !dst.empty();
    }

    static boolean readBgr(VideoCapture cap, Mat bgr) {
        if (cap == null || !cap.isOpened()) {
            return false;
        }
        Mat raw = new Mat();
        if (!cap.read(raw) || raw.empty()) {
            raw.release();
            return false;
        }
        boolean ok = toBgr(raw, bgr);
        raw.release();
        return ok && bgr != null && !bgr.empty();
    }

    static boolean readOrLoop(VideoCapture[] caps, int i, Path path, Mat frame) {
        if (readBgr(caps[i], frame)) {
            return true;
        }

        caps[i].set(Videoio.CAP_PROP_POS_FRAMES, 0);
        caps[i].set(Videoio.CAP_PROP_POS_MSEC, 0);
        if (readBgr(caps[i], frame)) {
            System.out.println("Video " + i + " looped via seek");
            return true;
        }

        caps[i].release();
        caps[i] = openVideo(path);
        if (readBgr(caps[i], frame)) {
            System.out.println("Video " + i + " reopened after loop");
            return true;
        }

        System.err.println("Warning: Could not read frame from video " + i);
        return false;
    }

    static byte[] encodeJpeg(Mat frame) {
        MatOfByte buf    = new MatOfByte();
        MatOfInt  params = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 88);
        Imgcodecs.imencode(".jpg", frame, buf, params);
        return buf.toArray();
    }

    static void writeFrame(OutputStream out, byte[] jpeg) throws IOException {
        String header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                      + jpeg.length + "\r\n\r\n";
        out.write(header.getBytes("UTF-8"));
        out.write(jpeg);
        out.write("\r\n".getBytes("UTF-8"));
        out.flush();
    }
}
