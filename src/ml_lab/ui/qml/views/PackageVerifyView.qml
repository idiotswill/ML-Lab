import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import "../components"

Item {
    id: root

    readonly property var packageVerify: appController.modelsRegistry.packageVerify

    FileDialog {
        id: exportDialog
        title: "Export release-candidate bundle"
        fileMode: FileDialog.SaveFile
        nameFilters: ["ZIP archive (*.zip)"]
        defaultSuffix: "zip"
        onAccepted: root.packageVerify.exportSelected(selectedFile)
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2

                Text {
                    text: "Package & Verify"
                    color: Theme.text
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }

                Text {
                    text: "Deterministic release bundles · fresh-process verification · immutable receipts"
                    color: Theme.muted
                    font.pixelSize: 12
                }
            }

            StatusPill {
                text: "INTEGRATION NO-GO"
                tone: "warn"
            }
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: operationText.implicitHeight + 20
            radius: 8
            color: root.packageVerify.busy ? Theme.warnSurface : Theme.surfaceAlt
            border.color: root.packageVerify.busy ? Theme.warnBorder : Theme.border

            Text {
                id: operationText
                anchors.fill: parent
                anchors.margins: 10
                text: root.packageVerify.busy ? root.packageVerify.busyMessage :
                      "A PASS receipt proves this bundle verified in a fresh process. It does not grant runtime integration authority."
                color: root.packageVerify.busy ? Theme.warnText : Theme.muted
                font.pixelSize: 12
                wrapMode: Text.WordWrap
                verticalAlignment: Text.AlignVCenter
            }
        }

        Panel {
            visible: !root.packageVerify.hasProject
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.centerIn: parent
                width: Math.min(parent.width - 48, 520)
                spacing: 10

                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "Choose a project first"
                    color: Theme.text
                    font.pixelSize: 20
                    font.weight: Font.DemiBold
                }

                Text {
                    Layout.fillWidth: true
                    text: "Packaging is project-scoped. Promote a registered model to RELEASE_CANDIDATE, then build and independently verify its immutable bundle here."
                    color: Theme.muted
                    font.pixelSize: 13
                    wrapMode: Text.WordWrap
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }

        SplitView {
            visible: root.packageVerify.hasProject
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            Panel {
                SplitView.preferredWidth: 230
                SplitView.minimumWidth: 160

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            Layout.fillWidth: true
                            text: "RELEASE CANDIDATES"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                        }

                        Text {
                            text: root.packageVerify.releaseCandidateRows.length
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                    }

                    ListView {
                        id: candidateList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 6
                        model: root.packageVerify.releaseCandidateRows

                        delegate: Rectangle {
                            required property var modelData
                            width: candidateList.width
                            height: 118
                            radius: 7
                            color: Theme.surfaceAlt
                            border.color: Theme.border

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 9
                                spacing: 3

                                RowLayout {
                                    Layout.fillWidth: true

                                    Text {
                                        Layout.fillWidth: true
                                        text: "Model " + modelData.shortId
                                        color: Theme.text
                                        font.pixelSize: 12
                                        font.weight: Font.DemiBold
                                    }

                                    StatusPill {
                                        text: modelData.stage
                                        tone: "good"
                                    }
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: "Experiment " + modelData.experimentShortId
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    elide: Text.ElideRight
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.modelSha256
                                    color: Theme.dim
                                    font.pixelSize: 9
                                    elide: Text.ElideMiddle
                                }

                                Item { Layout.fillHeight: true }

                                LabButton {
                                    Layout.fillWidth: true
                                    text: "Build bundle"
                                    primary: true
                                    enabled: !root.packageVerify.busy
                                    onClicked: root.packageVerify.buildBundle(modelData.id)
                                }
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: candidateList.count === 0
                            width: Math.min(parent.width - 24, 230)
                            text: "No RELEASE_CANDIDATE models. Promotion happens in Models & Registry."
                            color: Theme.dim
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                            horizontalAlignment: Text.AlignHCenter
                        }
                    }
                }
            }

            Panel {
                SplitView.preferredWidth: 250
                SplitView.minimumWidth: 180

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            Layout.fillWidth: true
                            text: "BUNDLE HISTORY"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                        }

                        Text {
                            text: root.packageVerify.bundleRows.length
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                    }

                    ListView {
                        id: bundleList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 5
                        model: root.packageVerify.bundleRows

                        delegate: ItemDelegate {
                            required property var modelData
                            width: bundleList.width
                            height: 82
                            activeFocusOnTab: true
                            Accessible.name: "Bundle " + modelData.shortId
                            onClicked: root.packageVerify.selectBundle(modelData.id)

                            contentItem: ColumnLayout {
                                spacing: 3

                                RowLayout {
                                    Layout.fillWidth: true

                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.shortId + " · model " + modelData.modelShortId
                                        color: Theme.text
                                        font.pixelSize: 12
                                        font.weight: Font.DemiBold
                                        elide: Text.ElideRight
                                    }

                                    StatusPill {
                                        text: "NO-GO"
                                        tone: "warn"
                                    }
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.bundleSha256
                                    color: Theme.dim
                                    font.pixelSize: 9
                                    elide: Text.ElideMiddle
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.createdAt
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    elide: Text.ElideRight
                                }
                            }

                            background: Rectangle {
                                radius: 7
                                color: root.packageVerify.selectedBundle.id === modelData.id ?
                                       Theme.accentSurface : (parent.hovered ? Theme.hover : "transparent")
                                border.color: parent.activeFocus ? Theme.accent :
                                              (root.packageVerify.selectedBundle.id === modelData.id ?
                                               Theme.accentBorder : "transparent")
                                border.width: parent.activeFocus ? 2 : 1
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: bundleList.count === 0
                            text: "No bundles built yet."
                            color: Theme.dim
                            font.pixelSize: 12
                        }
                    }
                }
            }

            Panel {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 260

                ScrollView {
                    anchors.fill: parent
                    clip: true

                    ColumnLayout {
                        width: Math.max(parent.width - 20, 240)
                        spacing: 10

                        Text {
                            text: root.packageVerify.selectedBundle.id ?
                                  "Bundle " + root.packageVerify.selectedBundle.shortId :
                                  "Bundle details"
                            color: Theme.text
                            font.pixelSize: 18
                            font.weight: Font.DemiBold
                        }

                        Text {
                            Layout.fillWidth: true
                            visible: !!root.packageVerify.selectedBundle.id
                            text: "Model " + (root.packageVerify.selectedBundle.modelShortId || "") +
                                  " · created " + (root.packageVerify.selectedBundle.createdAt || "")
                            color: Theme.muted
                            font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }

                        Text {
                            text: "BUNDLE SHA-256"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!root.packageVerify.selectedBundle.id
                        }

                        Text {
                            Layout.fillWidth: true
                            text: root.packageVerify.selectedBundle.bundleSha256 || "—"
                            color: Theme.text2
                            font.pixelSize: 10
                            wrapMode: Text.WrapAnywhere
                            visible: !!root.packageVerify.selectedBundle.id
                        }

                        Text {
                            text: "MANIFEST SHA-256"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!root.packageVerify.selectedBundle.id
                        }

                        Text {
                            Layout.fillWidth: true
                            text: root.packageVerify.selectedBundle.manifestSha256 || "—"
                            color: Theme.text2
                            font.pixelSize: 10
                            wrapMode: Text.WrapAnywhere
                            visible: !!root.packageVerify.selectedBundle.id
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            visible: !!root.packageVerify.selectedBundle.id

                            LabButton {
                                text: "Verify fresh"
                                primary: true
                                enabled: !root.packageVerify.busy
                                onClicked: root.packageVerify.verifySelected()
                            }

                            LabButton {
                                text: "Export ZIP"
                                enabled: !root.packageVerify.busy
                                onClicked: exportDialog.open()
                            }

                            Item { Layout.fillWidth: true }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: Theme.border
                            visible: !!root.packageVerify.selectedBundle.id
                        }

                        Text {
                            text: "VERIFICATION RECEIPTS"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                            visible: !!root.packageVerify.selectedBundle.id
                        }

                        Repeater {
                            model: root.packageVerify.receiptRows

                            delegate: Button {
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: 64
                                activeFocusOnTab: true
                                onClicked: root.packageVerify.selectReceipt(modelData.id)

                                contentItem: RowLayout {
                                    spacing: 8

                                    StatusPill {
                                        text: modelData.status
                                        tone: modelData.status === "PASS" ? "good" : "bad"
                                    }

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 2

                                        Text {
                                            Layout.fillWidth: true
                                            text: "Receipt " + modelData.shortId
                                            color: Theme.text
                                            font.pixelSize: 11
                                            font.weight: Font.DemiBold
                                            elide: Text.ElideRight
                                        }

                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.createdAt
                                            color: Theme.dim
                                            font.pixelSize: 9
                                            elide: Text.ElideRight
                                        }
                                    }
                                }

                                background: Rectangle {
                                    radius: 6
                                    color: parent.hovered ? Theme.hover : Theme.surfaceAlt
                                    border.color: parent.activeFocus ? Theme.accent : Theme.border
                                    border.width: parent.activeFocus ? 2 : 1
                                }
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: receiptPayload.implicitHeight + 18
                            radius: 6
                            color: Theme.surfaceAlt
                            border.color: Theme.border
                            visible: root.packageVerify.selectedReceiptPayload.length > 0

                            Text {
                                id: receiptPayload
                                anchors.fill: parent
                                anchors.margins: 9
                                text: root.packageVerify.selectedReceiptPayload
                                color: Theme.text2
                                font.pixelSize: 10
                                font.family: "Consolas"
                                wrapMode: Text.WrapAnywhere
                            }
                        }

                        Text {
                            Layout.fillWidth: true
                            visible: !root.packageVerify.selectedBundle.id
                            text: "Select a bundle to inspect hashes, run fresh verification, review immutable receipts, or export the verified artifact."
                            color: Theme.dim
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }
    }
}
