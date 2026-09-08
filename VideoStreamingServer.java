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
    private static final int OVERLAP_PX    = 56;
    /** Horizontal FOV of each rectified panel (cylindrical). Higher = more zoomed out. */
    private static final double OUTPUT_FOV_DEG = 128.0;
    /** Keep this fraction of the remapped frame (1.0 = no extra zoom crop). */
    private static final double CROP_WIDTH_FRACTION = 0.96;
    private static final double CROP_MAX_HEIGHT_FRACTION = 0.94;
    /** 0 = keep the top of the remap, 1 = keep the bottom (ground). */
    private static final double CROP_Y_BIAS = 0.50;
    /** Drop rows/cols darker than this after remap (fisheye rim / empty map). */
    private static final double VALID_LUMA_MIN = 14.0;
    /** Extra inset of the valid region so the curved fisheye rim is not stretched. */
    private static final double VALID_INSET_FRACTION = 0.015;
    /** Horizon row in the shared output frame (fraction of height from the top). */
    private static final double HORIZON_FRACTION = 0.40;
    /** Last stitch panel — rear camera (bumper at bottom of raw fisheye). */
    private static final int REAR_CAMERA_INDEX = 3;
    /**
     * Yaw of the right (index 2) and rear (index 3) panels toward their shared
     * seam. If the yellow van splits instead of merging, flip both signs.
     */
    private static final double RIGHT_SEAM_YAW_DEG = -6.0;
    private static final double REAR_SEAM_YAW_DEG  = 10.0;

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
          undistort[i] = FisheyePanelFilter.forStitchIndex(i);
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

    /**
     * Mount pose for one surround camera. Pitch is virtual-camera tilt in the
     * OpenCV Y-down frame: negative looks toward the top of the raw fisheye
     * (street instead of bumper when the lens points at the ground).
     */
    static final class PanelPose {
        final double pitchDeg;
        final double yawDeg;
        final double rollDeg;
        final double inputFovDeg;
        final double outputFovDeg;

        PanelPose(double pitchDeg, double yawDeg, double rollDeg,
                  double inputFovDeg, double outputFovDeg) {
            this.pitchDeg = pitchDeg;
            this.yawDeg = yawDeg;
            this.rollDeg = rollDeg;
            this.inputFovDeg = inputFovDeg;
            this.outputFovDeg = outputFovDeg;
        }
    }

    //  UNIFIED FISHEYE PANEL
    //
    //  Every feed is remapped with the same output size and FOV, then cropped
    //  to a filled rectangle (drops circular vignette). A small vertical crop
    //  nudge lines the horizons up; panels stay upright rectangles.

    static final class FisheyePanelFilter implements CameraFeedFilter {

        private static final double[] FISHEYE_D = { 0.0, 0.0, 0.0, 0.0 };
        private static final int WORK_HEIGHT = TARGET_HEIGHT * 2;
        private static final int WORK_WIDTH  = TARGET_WIDTH * 2;

        private final PanelPose pose;
        private Mat map1;
        private Mat map2;
        private Mat undistorted;
        private Rect workCrop;
        private int cachedSrcW = -1;
        private int cachedSrcH = -1;

        static FisheyePanelFilter forStitchIndex(int index) {
            final double inFov = 160.0;
            // Same pitch on every panel so objects keep the same height at seams.
            // Rear yaw swings the left edge toward the right-camera panel (index 2)
            // so the yellow van can occupy the overlap instead of two different scales.
            if (index == REAR_CAMERA_INDEX) {
                return new FisheyePanelFilter(new PanelPose(-6.0, REAR_SEAM_YAW_DEG, 0.0, inFov, OUTPUT_FOV_DEG));
            }
            if (index == REAR_CAMERA_INDEX - 1) {
                return new FisheyePanelFilter(new PanelPose(-6.0, RIGHT_SEAM_YAW_DEG, 0.0, inFov, OUTPUT_FOV_DEG));
            }
            return new FisheyePanelFilter(new PanelPose(-6.0, 0.0, 0.0, inFov, OUTPUT_FOV_DEG));
        }

        FisheyePanelFilter(PanelPose pose) {
            this.pose = pose;
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

            if (workCrop == null) {
                workCrop = horizonLockedCrop(undistorted);
            }
            Mat workRoi = undistorted.submat(workCrop);
            Imgproc.resize(workRoi, dst360x640, new Size(TARGET_WIDTH, TARGET_HEIGHT),
                    0, 0, Imgproc.INTER_AREA);
            workRoi.release();
            forceExactSize(dst360x640);
        }

        private void ensureMaps(int srcW, int srcH) {
            if (map1 != null && srcW == cachedSrcW && srcH == cachedSrcH) {
                return;
            }

            Mat K = equidistantK(srcW, srcH, pose.inputFovDeg);
            Mat D = distortionCoeffs();
            Mat R = eulerRyxz(pose.pitchDeg, pose.yawDeg, pose.rollDeg);

            if (map1 == null) map1 = new Mat();
            if (map2 == null) map2 = new Mat();

            // Cylindrical unwarp: azimuth is linear in pixels, so cars at the
            // panel edge (yellow van on the last feed) are not tan-stretched.
            buildCylindricalFisheyeMaps(K, D, R, srcW, srcH,
                    WORK_WIDTH, WORK_HEIGHT, pose.outputFovDeg, HORIZON_FRACTION,
                    map1, map2);

            cachedSrcW = srcW;
            cachedSrcH = srcH;
            workCrop = null;

            K.release();
            D.release();
            R.release();
        }

        static Mat equidistantK(int width, int height, double fovDeg) {
            double half = Math.toRadians(fovDeg) / 2.0;
            double f = (Math.min(width, height) / 2.0) / half;
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

        /**
         * Map a cylindrical panorama strip back into the fisheye. Horizontal
         * position is angle (not tan), which keeps a van at the edge of the
         * last panel closer to the same width as in the neighboring panel.
         */
        static void buildCylindricalFisheyeMaps(
                Mat K, Mat D, Mat R,
                int srcW, int srcH, int dstW, int dstH,
                double outputFovDeg, double horizonFraction,
                Mat map1, Mat map2) {

            double fx = K.get(0, 0)[0];
            double fy = K.get(1, 1)[0];
            double cx = K.get(0, 2)[0];
            double cy = K.get(1, 2)[0];
            double k1 = D.get(0, 0)[0];
            double k2 = D.get(1, 0)[0];
            double k3 = D.get(2, 0)[0];
            double k4 = D.get(3, 0)[0];

            Mat Rinv = new Mat();
            Core.invert(R, Rinv);
            double r00 = Rinv.get(0, 0)[0], r01 = Rinv.get(0, 1)[0], r02 = Rinv.get(0, 2)[0];
            double r10 = Rinv.get(1, 0)[0], r11 = Rinv.get(1, 1)[0], r12 = Rinv.get(1, 2)[0];
            double r20 = Rinv.get(2, 0)[0], r21 = Rinv.get(2, 1)[0], r22 = Rinv.get(2, 2)[0];
            Rinv.release();

            double half = Math.toRadians(outputFovDeg) / 2.0;
            double fCyl = (dstW / 2.0) / half;
            double cyOut = horizonFraction * dstH;

            Mat mapX = new Mat(dstH, dstW, CvType.CV_32FC1);
            Mat mapY = new Mat(dstH, dstW, CvType.CV_32FC1);
            float[] rowX = new float[dstW];
            float[] rowY = new float[dstW];

            for (int v = 0; v < dstH; v++) {
                double yc = (v - cyOut) / fCyl;
                for (int u = 0; u < dstW; u++) {
                    double theta = (u - dstW * 0.5) / fCyl;
                    double xv = Math.sin(theta);
                    double yv = yc;
                    double zv = Math.cos(theta);

                    double x = r00 * xv + r01 * yv + r02 * zv;
                    double y = r10 * xv + r11 * yv + r12 * zv;
                    double z = r20 * xv + r21 * yv + r22 * zv;

                    if (z <= 1e-6) {
                        rowX[u] = -1f;
                        rowY[u] = -1f;
                        continue;
                    }

                    double a = x / z;
                    double b = y / z;
                    double rho = Math.hypot(a, b);
                    double th = Math.atan(rho);
                    double th2 = th * th;
                    double th4 = th2 * th2;
                    double thd = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th4 * th2 + k4 * th4 * th4);
                    double scale = (rho > 1e-8) ? (thd / rho) : 1.0;
                    double su = fx * scale * a + cx;
                    double sv = fy * scale * b + cy;
                    if (su < -1 || sv < -1 || su > srcW || sv > srcH) {
                        rowX[u] = -1f;
                        rowY[u] = -1f;
                    } else {
                        rowX[u] = (float) su;
                        rowY[u] = (float) sv;
                    }
                }
                mapX.put(v, 0, rowX);
                mapY.put(v, 0, rowY);
            }

            Imgproc.convertMaps(mapX, mapY, map1, map2, CvType.CV_16SC2);
            mapX.release();
            mapY.release();
        }
    }


    static Rect cropWindow(Mat src) {
        Rect valid = validPixelRect(src);
        double aspect = (double) TARGET_WIDTH / TARGET_HEIGHT;
        int imgW = src.cols();
        int imgH = src.rows();

        int cropW = Math.max(2, (int) Math.round(imgW * CROP_WIDTH_FRACTION));
        int cropH = Math.max(2, (int) Math.round(cropW / aspect));
        if (cropH > imgH * CROP_MAX_HEIGHT_FRACTION) {
            cropH = Math.max(2, (int) Math.round(imgH * CROP_MAX_HEIGHT_FRACTION));
            cropW = Math.max(2, (int) Math.round(cropH * aspect));
        }

        cropW = Math.min(cropW, valid.width);
        cropH = Math.min(cropH, (int) Math.round(cropW / aspect));
        if (cropH > valid.height) {
            cropH = valid.height;
            cropW = Math.max(2, (int) Math.round(cropH * aspect));
            if (cropW > valid.width) {
                cropW = valid.width;
                cropH = Math.max(2, (int) Math.round(cropW / aspect));
            }
        }

        int x = valid.x + (valid.width - cropW) / 2;
        int y = valid.y + (int) Math.round((valid.height - cropH) * CROP_Y_BIAS);
        x = Math.max(0, Math.min(x, imgW - cropW));
        y = Math.max(0, Math.min(y, imgH - cropH));
        return new Rect(x, y, cropW, cropH);
    }

    /**
     * Same filled 16:9 window as {@link #cropWindow}, then a small vertical
     * nudge so the estimated horizon sits near HORIZON_FRACTION. No rotation:
     * each panel stays an upright rectangle.
     */
    static Rect horizonLockedCrop(Mat src) {
        Rect crop = cropWindow(src);
        Mat preview = new Mat();
        Mat roi = src.submat(crop);
        Imgproc.resize(roi, preview, new Size(TARGET_WIDTH, TARGET_HEIGHT),
                0, 0, Imgproc.INTER_AREA);
        roi.release();

        int horizon = estimateHorizonRow(preview);
        preview.release();

        int targetY = (int) Math.round(HORIZON_FRACTION * TARGET_HEIGHT);
        double dyPanel = targetY - horizon;
        int dyWork = (int) Math.round(dyPanel * ((double) crop.height / TARGET_HEIGHT));
        int maxShift = (int) Math.round(crop.height * 0.12);
        if (dyWork > maxShift) dyWork = maxShift;
        if (dyWork < -maxShift) dyWork = -maxShift;

        int y = crop.y - dyWork;
        y = Math.max(0, Math.min(y, src.rows() - crop.height));
        return new Rect(crop.x, y, crop.width, crop.height);
    }

    /** Filled pixels only, inset so the circular fisheye rim is not stretched. */
    static Rect validPixelRect(Mat bgr) {
        int imgW = bgr.cols();
        int imgH = bgr.rows();
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Mat mask = new Mat();
        Imgproc.threshold(gray, mask, VALID_LUMA_MIN, 255, Imgproc.THRESH_BINARY);

        int k = Math.max(7, Math.min(imgW, imgH) / 48);
        if ((k & 1) == 0) {
            k++;
        }
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(k, k));
        Imgproc.erode(mask, mask, kernel);

        Rect box = Imgproc.boundingRect(mask);
        gray.release();
        mask.release();
        kernel.release();

        if (box.width < imgW / 5 || box.height < imgH / 5) {
            int padX = (int) Math.round(imgW * 0.08);
            int padY = (int) Math.round(imgH * 0.10);
            return new Rect(padX, padY, Math.max(2, imgW - 2 * padX), Math.max(2, imgH - 2 * padY));
        }

        int insetX = Math.max(2, (int) Math.round(box.width * VALID_INSET_FRACTION));
        int insetY = Math.max(2, (int) Math.round(box.height * VALID_INSET_FRACTION));
        int x = box.x + insetX;
        int y = box.y + insetY;
        int w = box.width - 2 * insetX;
        int h = box.height - 2 * insetY;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + w > imgW) w = imgW - x;
        if (y + h > imgH) h = imgH - y;
        if (w < 2 || h < 2) {
            return new Rect(0, 0, imgW, imgH);
        }
        return new Rect(x, y, w, h);
    }

    static int estimateHorizonRow(Mat bgr) {
        Mat gray = new Mat();
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(9, 9), 1.4);
        Mat sobel = new Mat();
        Imgproc.Sobel(gray, sobel, CvType.CV_32F, 0, 1, 3);
        Mat mag = new Mat();
        Core.convertScaleAbs(sobel, mag);
        sobel.release();
        sobel = mag;

        int h = gray.rows();
        int w = gray.cols();
        int x0 = w / 5;
        int x1 = w - w / 5;
        int y0 = Math.max(1, (int) (h * 0.08));
        int y1 = Math.max(y0 + 1, (int) (h * 0.82));
        int fallback = (int) Math.round(HORIZON_FRACTION * h);

        double best = -1;
        int bestY = fallback;
        for (int y = y0; y < y1; y++) {
            double s = Core.sumElems(sobel.row(y).colRange(x0, x1)).val[0];
            if (s > best) {
                best = s;
                bestY = y;
            }
        }
        gray.release();
        sobel.release();
        return bestY;
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
    //  Upright rectangular panels, feather-blended into one strip.

    static Mat featherStitch(Mat[] frames) {
        int N = frames.length;
        int H = TARGET_HEIGHT;
        int W = TARGET_WIDTH;

        int overlap = Math.min(OVERLAP_PX, W / 4);
        int panoW   = W + (N - 1) * (W - overlap);

        Mat accumColor  = Mat.zeros(H, panoW, CvType.CV_32FC3);
        Mat accumWeight = Mat.zeros(H, panoW, CvType.CV_32FC1);

        for (int i = 0; i < N; i++) {
            forceExactSize(frames[i]);
            int xStart = i * (W - overlap);

            Mat weight = buildFeatherMask(H, W, overlap, i > 0, i < N - 1);

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
