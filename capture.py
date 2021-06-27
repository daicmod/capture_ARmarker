
import cv2
import numpy as np

aruco = cv2.aruco  # arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
camera_matrix = np.array([[639.87721705,   0., 330.12073612],
                          [0., 643.69687408, 208.61588364],
                          [0.,   0.,   1.]])
distortion_coeff = np.array(
    [5.66942769e-02, -6.05774927e-01, -7.42066667e-03, -3.09571466e-04, 1.92386974e+00])
marker_length = 0.04  # [m]
target_width = 0.15
target_height = 0.05


def getGrayImage(frame):
    # グレースケール化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def getBinaryImage(frame):
    # 単純二値化
    ret, img_binary = cv2.threshold(frame,
                                    60, 255,
                                    cv2.THRESH_BINARY)
    return ret, img_binary


def estimatePoseSingleRectangle(coners, targetWidth, targetHeight, _cameraMatrix, _distCoeffs):
    assert(targetWidth > 0 and targetHeight > 0)
    # _getSingleMarkerObjectPoints
    makerObjectPoints = np.array(
        [[-targetWidth / 2.0,  targetHeight / 2.0, 0],
         [targetWidth / 2.0,  targetHeight / 2.0, 0],
         [targetWidth / 2.0, -targetHeight / 2.0, 0],
         [-targetWidth / 2.0, -targetHeight / 2.0, 0]])

    # create(nMarkers, 1, CV_64FC3)
    err, rvecs, tvecs = cv2.solvePnP(
        makerObjectPoints, coners, _cameraMatrix, _distCoeffs)

    return rvecs, tvecs, makerObjectPoints


def update(capture):
    while(True):
        ret, frame = capture.read()
        windowsize = (frame.shape[1], frame.shape[0])
        frame = cv2.resize(frame, windowsize)

        # step1. マーカー検出
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            frame, dictionary)
        aruco.drawDetectedMarkers(
            frame, corners, ids, (0, 255, 0))  # 検出したマーカに描画する

        # マーカーが1つ以上あったときは処理する
        if len(corners) > 0:
            id1corner = []
            id2corner = []
            id3corner = []
            id4corner = []
            # 全てのマーカーを確認して、id1とid4を格納する。なければスキップ。
            for i, corner in enumerate(corners):
                if ids[i][0] == 2:
                    id2corner = corners[i][0]
                    continue
                if ids[i][0] == 3:
                    id3corner = corners[i][0]
                    continue
                if ids[i][0] == 4:
                    id4corner = corners[i][0]
                    continue
                if ids[i][0] == 1:
                    id1corner = corners[i][0]
                rvec2, tvec2, _2 = aruco.estimatePoseSingleMarkers(
                    corner, marker_length, camera_matrix, distortion_coeff)

                # 不要なaxisを除去
                tvec2 = np.squeeze(tvec2)
                rvec2 = np.squeeze(rvec2)
                # 回転ベクトルからrodoriguesへ変換
                rvec_matrix = cv2.Rodrigues(rvec2)
                rvec_matrix = rvec_matrix[0]  # rodoriguesから抜き出し
                # 並進ベクトルの転置
                transpose_tvec = tvec2[np.newaxis, :].T
                # 合成
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                # オイラー角への変換
                euler_angle2 = cv2.decomposeProjectionMatrix(proj_matrix)[
                    6]  # [deg]

                # 可視化
                draw_pole_length = marker_length/2  # 現実での長さ[m]
                aruco.drawAxis(frame, camera_matrix, distortion_coeff,
                               rvec2, tvec2, draw_pole_length)

            if len(id1corner) == 0 or len(id4corner) == 0:
                continue

            # マーカーで囲んだ区間を指定して画像を切り抜く
            y1 = int(id1corner[2][1])
            y2 = int(id4corner[0][1])
            x1 = int(id1corner[0][0])
            x2 = int(id4corner[1][0])
            if x2 > frame.shape[1]:
                x2 = frame.shape[1]
            if y2 > frame.shape[0]:
                y2 = frame.shape[0]
            if x2 - x1 < 100:
                break
            if y2 - y1 < 100:
                break
            frame2 = frame[y1:y2, x1:x2]

            # step2. グレースケール化
            img_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # step3. 単純二値化
            ret, img_binary = cv2.threshold(img_gray,
                                            200, 255,
                                            cv2.THRESH_BINARY_INV)

            # step4. 輪郭抽出
            contours, hierarchy = cv2.findContours(img_binary,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            # step5. 抽出した輪郭のうち最も面積の大きい輪郭をターゲットと設定する
            max_area = 0
            max_contour = contours[0]
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            # step.6 ターゲット輪郭を直線近似した輪郭に変形する
            arclen = cv2.arcLength(max_contour,
                                   True)  # 対象領域が閉曲線の場合、True
            approx = cv2.approxPolyDP(max_contour,
                                      0.1*arclen,  # 近似の具合?
                                      True)
            if len(approx) == 4:
                img_contour = cv2.drawContours(
                    frame2, [approx], -1, (0, 255, 0), 1)
                corner[0][0][0] = approx[0][0][0]
                corner[0][0][1] = approx[0][0][1]
                corner[0][1][0] = approx[1][0][0]
                corner[0][1][1] = approx[1][0][1]
                corner[0][2][0] = approx[2][0][0]
                corner[0][2][1] = approx[2][0][1]
                corner[0][3][0] = approx[3][0][0]
                corner[0][3][1] = approx[3][0][1]

                rvec, tvec, _ = estimatePoseSingleRectangle(
                    corner, target_width, target_height, camera_matrix, distortion_coeff)

                # 不要なaxisを除去
                tvec = np.squeeze(tvec)
                rvec = np.squeeze(rvec)
                # 回転ベクトルからrodoriguesへ変換
                rvec_matrix = cv2.Rodrigues(rvec)
                rvec_matrix = rvec_matrix[0]  # rodoriguesから抜き出し
                # 並進ベクトルの転置
                transpose_tvec = tvec[np.newaxis, :].T
                # 合成
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                # オイラー角への変換
                euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[
                    6]  # [deg]

                # 可視化
                draw_pole_length = target_height/2  # 現実での長さ[m]
                aruco.drawAxis(frame2, camera_matrix, distortion_coeff,
                               rvec, tvec, draw_pole_length)

                print('maker  ' +
                      'x :' + str(tvec2[0]*100) +
                      '\ty :' + str(tvec2[1]*100) +
                      '\tz :' + str(tvec2[2]*100) +
                      '\troll :' + str(euler_angle2[0]) +
                      '\tpitch :' + str(euler_angle2[1]) +
                      '\tyaw :' + str(euler_angle2[2]))

                print('target ' +
                      'x :' + str(tvec[0]*100) +
                      '\ty :' + str(tvec[1]*100) +
                      '\tz :' + str(tvec[2]*100) +
                      '\troll :' + str(euler_angle[0]) +
                      '\tpitch :' + str(euler_angle[1]) +
                      '\tyaw :' + str(euler_angle[2]))

            cv2.imshow('frame2', frame2)
            cv2.imshow('img_binary', img_binary)

        cv2.imshow('output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    capture = cv2.VideoCapture(2)

    update(capture)

    capture.release()
    cv2.destroyAllWindows()
