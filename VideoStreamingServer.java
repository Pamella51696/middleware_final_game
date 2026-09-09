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

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

public class VideoStreamingServer {

    private static final int DEFAULT_PORT  = 9090;
    private static final int TARGET_HEIGHT = 360;
    private static final int TARGET_WIDTH  = 640;
    private static final int PANEL_WIDTH   = TARGET_WIDTH;
    private static final int PANEL_HEIGHT  = TARGET_HEIGHT;
    /** Extra black band above/below the panorama strip. */
    private static final int CANVAS_PAD_Y  = 16;
    /**
     * Horizontal span of each spherical panel (deg). Wider than 90° overlaps
     * neighbors so the same world rays can blend. Spacing is 90° (4 cameras).
     */
    private static final double PANEL_YAW_DEG = 120.0;
    /** Vertical span of each spherical panel (deg). */
    private static final double PANEL_PITCH_DEG = 80.0;
    /** Yaw of Left, Front, Right, Rear in the vehicle frame (0° = forward). */
    private static final double[] CAM_YAW_DEG = { -90.0, 0.0, 90.0, 180.0 };
    /** Pitch: negative looks toward the ground. */
    private static final double[] CAM_PITCH_DEG = { -6.0, -6.0, -6.0, -6.0 };
    private static final double[] CAM_ROLL_DEG  = { 0.0, 0.0, 0.0, 0.0 };
    /** Horizon row in the shared output frame (fraction of height from the top). */
    private static final double HORIZON_FRACTION = 0.40;

    /** Approximate fisheye FOV. Replace with calibrated value later. */
    private static final double INPUT_FISHEYE_FOV_DEG = 180.0;

    /**
     * Maximum angle sampled from the fisheye optical axis.
     *
     * IMPORTANT:
     * Start with 88 degrees while debugging.
     * 76 degrees throws away too much of the outer fisheye image.
     */
    private static final double MAX_INCIDENCE_DEG = 88.0;

    /**
     * Fisheye optical center.
     *
     * For the current image this is initially the image center.
     * These MUST eventually come from camera calibration.
     */
    private static final double FISHEYE_CX = 0.50;
    private static final double FISHEYE_CY = 0.50;

    /**
     * Fisheye focal length in pixels. 0 = derive from INPUT_FISHEYE_FOV_DEG
     * and the source size. Set from calibration when available.
     */
    private static final double FISHEYE_FX = 0.0;
    private static final double FISHEYE_FY = 0.0;

    /**
     * Fisheye distortion coefficients.
     *
     * OpenCV fisheye model:
     *
     * theta_d =
     *     theta
     *   + k1*theta^3
     *   + k2*theta^5
     *   + k3*theta^7
     *   + k4*theta^9
     *
     * PLACEHOLDERS until the camera is calibrated. Do not guess values.
     */
    private static final double[] FISHEYE_K = {
        0.0,   // k1
        0.0,   // k2
        0.0,   // k3
        0.0    // k4
    };

    // =========================================================================
    public static void main(String[] args) throws IOException {

        Path leftVideo  = ensureDecodable(Paths.get("left_1.mp4"));
        Path frontVideo = ensureDecodable(Paths.get("front_1.mp4"));
        Path rightVideo = ensureDecodable(Paths.get("right_1.mp4"));
        Path backVideo  = ensureDecodable(Paths.get("rear_1.mp4"));

        if (args.length >= 5) {
            leftVideo  = ensureDecodable(Paths.get(args[1]));
            frontVideo = ensureDecodable(Paths.get(args[2]));
            rightVideo = ensureDecodable(Paths.get(args[3]));
            backVideo  = ensureDecodable(Paths.get(args[4]));
        }

        Path[] videos = { leftVideo, frontVideo, rightVideo, backVideo };

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
        System.out.println("Feeds: left=" + leftVideo + " front=" + frontVideo
                + " right=" + rightVideo + " rear=" + backVideo);
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
          undistort[i] = SphericalPanel.forStitchIndex(i);
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

            int[] horizons = new int[ready.length];
            for (int i = 0; i < ready.length; i++) {
              horizons[i] = undistort[i].horizonRow();
            }

            Mat panorama = featherStitch(ready, horizons);
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
        /** Horizon row in the filtered TARGET-sized panel (pixels from the top). */
        int horizonRow();
    }


    //  SPHERICAL PANEL
    //
    //  Each camera samples a longitude window of a shared sphere, then those
    //  world rays are projected into that camera's fisheye. Adjacent panels
    //  overlap in yaw so the same directions can blend. Horizon is locked by
    //  construction (HORIZON_FRACTION). Empty samples stay black.

    static final class SphericalPanel implements CameraFeedFilter {

        private final int index;
        private final double[] R;
        private Mat map1;
        private Mat map2;
        private Mat undistorted;
        private int cachedSrcW = -1;
        private int cachedSrcH = -1;

        static SphericalPanel forStitchIndex(int index) {
            if (index < 0 || index >= CAM_YAW_DEG.length) {
                index = Math.max(0, Math.min(index, CAM_YAW_DEG.length - 1));
            }
            return new SphericalPanel(index);
        }

        SphericalPanel(int index) {
            this.index = index;
            this.R = worldToCameraR(
                    CAM_PITCH_DEG[index],
                    CAM_YAW_DEG[index],
                    CAM_ROLL_DEG[index]);
        }

        @Override
        public void recalibrateAndFilter(Mat src, Mat dst360x640) {
            if (src == null || src.empty()) {
                return;
            }

            ensureMaps(src.cols(), src.rows());

            if (undistorted == null) undistorted = new Mat();

            Imgproc.remap(src, undistorted, map1, map2, Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT);
            if (undistorted.cols() == TARGET_WIDTH && undistorted.rows() == TARGET_HEIGHT) {
                undistorted.copyTo(dst360x640);
            } else {
                Imgproc.resize(undistorted, dst360x640,
                        new Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, Imgproc.INTER_AREA);
            }
            forceExactSize(dst360x640);
        }

        @Override
        public int horizonRow() {
            return (int) Math.round(HORIZON_FRACTION * PANEL_HEIGHT);
        }

        private void ensureMaps(int srcW, int srcH) {
            if (map1 != null && srcW == cachedSrcW && srcH == cachedSrcH) {
                return;
            }

            double f = fisheyeFocal(srcW, srcH, INPUT_FISHEYE_FOV_DEG);
            double fx = FISHEYE_FX > 0.0 ? FISHEYE_FX : f;
            double fy = FISHEYE_FY > 0.0 ? FISHEYE_FY : f;
            double cx = srcW * FISHEYE_CX;
            double cy = srcH * FISHEYE_CY;

            double yaw0 = Math.toRadians(CAM_YAW_DEG[index]);
            double yawSpan = Math.toRadians(PANEL_YAW_DEG);
            double pitchSpan = Math.toRadians(PANEL_PITCH_DEG);

            double horizonY = HORIZON_FRACTION * PANEL_HEIGHT;

            double fyOut =
                    (PANEL_HEIGHT / 2.0)
                    / Math.tan(pitchSpan / 2.0);

            double maxInc =
                    Math.toRadians(MAX_INCIDENCE_DEG);

            Mat mapX =
                    new Mat(PANEL_HEIGHT, PANEL_WIDTH, CvType.CV_32FC1);

            Mat mapY =
                    new Mat(PANEL_HEIGHT, PANEL_WIDTH, CvType.CV_32FC1);

            float[] rowX = new float[PANEL_WIDTH];
            float[] rowY = new float[PANEL_WIDTH];

            for (int v = 0; v < PANEL_HEIGHT; v++) {

                double phi =
                        Math.atan((horizonY - (v + 0.5)) / fyOut);

                double cphi = Math.cos(phi);
                double sphi = Math.sin(phi);

                for (int u = 0; u < PANEL_WIDTH; u++) {

                    double theta =
                            yaw0
                            + ((u + 0.5) / PANEL_WIDTH - 0.5)
                            * yawSpan;

                    // Z-up world sphere: theta = yaw, phi = elevation.
                    double xv = cphi * Math.cos(theta);
                    double yv = cphi * Math.sin(theta);
                    double zv = sphi;

                    double xc =
                            R[0] * xv +
                            R[3] * yv +
                            R[6] * zv;

                    double yc =
                            R[1] * xv +
                            R[4] * yv +
                            R[7] * zv;

                    double zc =
                            R[2] * xv +
                            R[5] * yv +
                            R[8] * zv;

                    if (zc <= 1e-4) {
                        rowX[u] = -1f;
                        rowY[u] = -1f;
                        continue;
                    }

                    double inc =
                            Math.atan2(
                                    Math.hypot(xc, yc),
                                    zc
                            );

                    if (inc > maxInc) {
                        rowX[u] = -1f;
                        rowY[u] = -1f;
                        continue;
                    }

                    /*
                     * Convert the 3D incidence angle into a REAL fisheye
                     * image radius (OpenCV theta_d * focal).
                     */
                    double thetaD = fisheyeThetaDistorted(inc);
                    double az = Math.atan2(yc, xc);

                    float su = (float) (cx + fx * thetaD * Math.cos(az));
                    float sv = (float) (cy + fy * thetaD * Math.sin(az));

                    if (su < 1 ||
                        sv < 1 ||
                        su >= srcW - 1 ||
                        sv >= srcH - 1) {

                        rowX[u] = -1f;
                        rowY[u] = -1f;

                    } else {

                        rowX[u] = su;
                        rowY[u] = sv;
                    }
                }

                mapX.put(v, 0, rowX);
                mapY.put(v, 0, rowY);
            }

            if (map1 == null)
                map1 = new Mat();

            if (map2 == null)
                map2 = new Mat();

            Imgproc.convertMaps(
                    mapX,
                    mapY,
                    map1,
                    map2,
                    CvType.CV_16SC2,
                    false
            );

            mapX.release();
            mapY.release();

            cachedSrcW = srcW;
            cachedSrcH = srcH;
        }
    }

    /**
     * World (Z-up) to camera (X right, Y down, Z forward) rotation, stored
     * column-major as R[0..8] for {@code cam = R * world}.
     */
    static double[] worldToCameraR(double pitchDeg, double yawDeg, double rollDeg) {
        double yaw = Math.toRadians(yawDeg);
        double pitch = Math.toRadians(pitchDeg);
        double roll = Math.toRadians(rollDeg);

        double cy = Math.cos(yaw), sy = Math.sin(yaw);
        double cp = Math.cos(pitch), sp = Math.sin(pitch);
        double cr = Math.cos(roll), sr = Math.sin(roll);

        double zx = cp * cy, zy = cp * sy, zz = sp;
        double xx0 = sy, xy0 = -cy, xz0 = 0.0;
        double yx0 = zy * xz0 - zz * xy0;
        double yy0 = zz * xx0 - zx * xz0;
        double yz0 = zx * xy0 - zy * xx0;

        double xx = cr * xx0 + sr * yx0;
        double xy = cr * xy0 + sr * yy0;
        double xz = cr * xz0 + sr * yz0;
        double yx = -sr * xx0 + cr * yx0;
        double yy = -sr * xy0 + cr * yy0;
        double yz = -sr * xz0 + cr * yz0;

        return new double[] { xx, yx, zx, xy, yy, zy, xz, yz, zz };
    }

    static double fisheyeFocal(int width, int height, double fovDeg) {
        double half = Math.toRadians(fovDeg) / 2.0;
        return (Math.min(width, height) / 2.0) / half;
    }

    static double fisheyeThetaDistorted(double incidence) {
        double theta = incidence;
        double theta2 = theta * theta;
        double theta3 = theta2 * theta;
        double theta5 = theta3 * theta2;
        double theta7 = theta5 * theta2;
        double theta9 = theta7 * theta2;
        return theta
                + FISHEYE_K[0] * theta3
                + FISHEYE_K[1] * theta5
                + FISHEYE_K[2] * theta7
                + FISHEYE_K[3] * theta9;
    }

    static double fisheyeRadius(double incidence, double focal) {
        return focal * fisheyeThetaDistorted(incidence);
    }

    /** Pixel overlap implied by PANEL_YAW_DEG vs 90° camera spacing. */
    static int geometricOverlapPx() {
        double spacing = 360.0 / 4.0;
        double overlapDeg = PANEL_YAW_DEG - spacing;
        if (overlapDeg < 0.0) {
            overlapDeg = 0.0;
        }
        int px = (int) Math.round(PANEL_WIDTH * overlapDeg / PANEL_YAW_DEG);
        return Math.max(0, Math.min(px, PANEL_WIDTH / 4));
    }

    static void forceExactSize(Mat img) {
        if (img.cols() == TARGET_WIDTH && img.rows() == TARGET_HEIGHT) {
            return;
        }
        Mat tmp = new Mat();
        Imgproc.resize(img, tmp, new Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, Imgproc.INTER_AREA);
        tmp.copyTo(img);
        tmp.release();
    }


    //  CORE BLENDING  —  featherStitch
    //  Spherical panels share one horizon. Overlap width matches the extra
    //  yaw beyond 90° so neighbors blend the same world rays. Unused canvas
    //  stays black.

    static Mat featherStitch(Mat[] frames) {
        int[] horizons = new int[frames.length];
        for (int i = 0; i < frames.length; i++) {
            horizons[i] = (int) Math.round(HORIZON_FRACTION * TARGET_HEIGHT);
        }
        return featherStitch(frames, horizons);
    }

    static Mat featherStitch(Mat[] frames, int[] horizons) {
        int N = frames.length;
        int H = TARGET_HEIGHT;
        int W = TARGET_WIDTH;

        int overlap = geometricOverlapPx();
        int panoW   = W + (N - 1) * (W - overlap);

        int minHor = H;
        int maxHor = 0;
        int[] hor = new int[N];
        for (int i = 0; i < N; i++) {
            int h = (horizons != null && i < horizons.length) ? horizons[i]
                    : (int) Math.round(HORIZON_FRACTION * H);
            if (h < 1) h = 1;
            if (h > H - 2) h = H - 2;
            hor[i] = h;
            minHor = Math.min(minHor, h);
            maxHor = Math.max(maxHor, h);
        }
        int pad = Math.max(0, CANVAS_PAD_Y);
        int panoH = H + (maxHor - minHor) + 2 * pad;
        int commonHorizonY = maxHor + pad;

        Mat accumColor  = Mat.zeros(panoH, panoW, CvType.CV_32FC3);
        Mat accumWeight = Mat.zeros(panoH, panoW, CvType.CV_32FC1);

        for (int i = 0; i < N; i++) {
            forceExactSize(frames[i]);
            int xStart = i * (W - overlap);
            int yStart = commonHorizonY - hor[i];

            Mat weight = buildFeatherMask(H, W, overlap, i > 0, i < N - 1);

            Mat frameF = new Mat();
            frames[i].convertTo(frameF, CvType.CV_32FC3);

            Mat weight3 = new Mat();
            List<Mat> ch = new ArrayList<>();
            ch.add(weight); ch.add(weight); ch.add(weight);
            Core.merge(ch, weight3);

            Mat wFrame = new Mat();
            Core.multiply(frameF, weight3, wFrame);

            int srcX = 0;
            int srcY = 0;
            if (xStart < 0) {
                srcX = -xStart;
                xStart = 0;
            }
            if (yStart < 0) {
                srcY = -yStart;
                yStart = 0;
            }
            int xEnd = Math.min(xStart + (W - srcX), panoW);
            int yEnd = Math.min(yStart + (H - srcY), panoH);
            int wActual = xEnd - xStart;
            int hActual = yEnd - yStart;
            if (wActual <= 0 || hActual <= 0) {
                frameF.release(); weight.release(); weight3.release();
                wFrame.release();
                continue;
            }

            Mat colorRoi  = accumColor.submat(yStart, yEnd, xStart, xEnd);
            Mat weightRoi = accumWeight.submat(yStart, yEnd, xStart, xEnd);

            Mat wFrameCrop = wFrame.rowRange(srcY, srcY + hActual).colRange(srcX, srcX + wActual);
            Mat weightCrop = weight.rowRange(srcY, srcY + hActual).colRange(srcX, srcX + wActual);

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

    static Mat buildFeatherMask(int H, int W, int overlap, boolean fadeLeft, boolean fadeRight) {
        Mat mask = new Mat(H, W, CvType.CV_32FC1, new Scalar(1.0));
        if (overlap <= 0) {
            return mask;
        }
        for (int x = 0; x < overlap; x++) {
            float alpha = (float) x / overlap;
            if (fadeLeft) {
                Mat colL = mask.col(x);
                colL.setTo(new Scalar(alpha));
                colL.release();
            }
            if (fadeRight) {
                Mat colR = mask.col(W - 1 - x);
                colR.setTo(new Scalar(alpha));
                colR.release();
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
