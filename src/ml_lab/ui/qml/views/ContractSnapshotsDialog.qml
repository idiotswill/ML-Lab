import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import "../components"

Dialog {
    id: root

    readonly property var contracts: appController.modelsRegistry.contractSnapshots
    readonly property var snapshot: contracts.selectedSnapshot

    parent: Overlay.overlay
    modal: true
    width: Math.min(parent ? parent.width - 48 : 920, 1040)
    height: Math.min(parent ? parent.height - 48 : 620, 680)
    anchors.centerIn: parent
    title: "Contract snapshots"
    standardButtons: Dialog.NoButton

    background: Rectangle {
        color: Theme.surface
        radius: 12
        border.color: Theme.borderStrong
    }

    FolderDialog {
        id: repositoryDialog
        title: "Choose local Git checkout"
        onAccepted: repositoryPath.text = selectedFolder
    }

    contentItem: ColumnLayout {
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2

                Text {
                    text: "Contract provenance"
                    color: Theme.text
                    font.pixelSize: 21
                    font.weight: Font.DemiBold
                }

                Text {
                    Layout.fillWidth: true
                    text: contracts.captureMessage
                    color: Theme.muted
                    font.pixelSize: 11
                    wrapMode: Text.WordWrap
                }
            }

            StatusPill {
                text: "READ-ONLY GIT"
                tone: "neutral"
            }

            StatusPill {
                text: "INTEGRATION NO-GO"
                tone: "warn"
            }
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: captureColumn.implicitHeight + 18
            radius: 8
            color: contracts.busy ? Theme.warnSurface : Theme.surfaceAlt
            border.color: contracts.busy ? Theme.warnBorder : Theme.border

            ColumnLayout {
                id: captureColumn
                anchors.fill: parent
                anchors.margins: 9
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true

                    TextField {
                        id: repositoryPath
                        Layout.fillWidth: true
                        placeholderText: "Local Frankenhomie Git checkout"
                        text: contracts.suggestedRepository
                        enabled: contracts.captureSupported && !contracts.busy
                        selectByMouse: true
                    }

                    LabButton {
                        text: "Browse…"
                        enabled: contracts.captureSupported && !contracts.busy
                        onClicked: repositoryDialog.open()
                    }

                    TextField {
                        id: gitRef
                        Layout.preferredWidth: 150
                        text: "master"
                        placeholderText: "Git ref"
                        enabled: contracts.captureSupported && !contracts.busy
                        selectByMouse: true
                    }

                    LabButton {
                        text: "Freeze snapshot"
                        primary: true
                        enabled: contracts.captureSupported && !contracts.busy &&
                                 repositoryPath.text.length > 0 && gitRef.text.trim().length > 0
                        onClicked: contracts.capture(repositoryPath.text, gitRef.text)
                    }
                }

                Text {
                    Layout.fillWidth: true
                    text: contracts.busy ? contracts.busyMessage :
                          "Capture resolves the ref to a full commit and reads only committed bytes. Dirty working-tree files are recorded as provenance but never included."
                    color: contracts.busy ? Theme.warnText : Theme.dim
                    font.pixelSize: 10
                    wrapMode: Text.WordWrap
                }
            }
        }

        SplitView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            orientation: Qt.Horizontal

            Panel {
                SplitView.preferredWidth: 285
                SplitView.minimumWidth: 220

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            Layout.fillWidth: true
                            text: "SNAPSHOT HISTORY"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                        }

                        Text {
                            text: contracts.snapshotRows.length
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                    }

                    ListView {
                        id: snapshotList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 5
                        model: contracts.snapshotRows

                        delegate: ItemDelegate {
                            required property var modelData
                            width: snapshotList.width
                            height: 82
                            activeFocusOnTab: true
                            Accessible.name: "Contract snapshot " + modelData.commitShort
                            onClicked: contracts.selectSnapshot(modelData.id)

                            contentItem: ColumnLayout {
                                spacing: 3

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.commitShort + " · " + modelData.contractVersion
                                    color: Theme.text
                                    font.pixelSize: 11
                                    font.weight: Font.DemiBold
                                    elide: Text.ElideRight
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.compatibilitySignature
                                    color: Theme.dim
                                    font.pixelSize: 9
                                    elide: Text.ElideMiddle
                                }

                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.createdAt
                                    color: Theme.muted
                                    font.pixelSize: 9
                                    elide: Text.ElideRight
                                }
                            }

                            background: Rectangle {
                                radius: 7
                                color: root.snapshot.id === modelData.id ?
                                       Theme.accentSurface : (parent.hovered ? Theme.hover : "transparent")
                                border.color: parent.activeFocus ? Theme.accent :
                                              (root.snapshot.id === modelData.id ?
                                               Theme.accentBorder : "transparent")
                                border.width: parent.activeFocus ? 2 : 1
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: snapshotList.count === 0
                            width: Math.min(parent.width - 20, 230)
                            text: contracts.captureSupported ?
                                  "No contract snapshots yet." :
                                  "This adapter does not declare a contract snapshot boundary."
                            color: Theme.dim
                            font.pixelSize: 11
                            wrapMode: Text.WordWrap
                            horizontalAlignment: Text.AlignHCenter
                        }
                    }
                }
            }

            Panel {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 380

                ScrollView {
                    anchors.fill: parent
                    clip: true

                    ColumnLayout {
                        width: Math.max(parent.width - 20, 350)
                        spacing: 8

                        Text {
                            text: root.snapshot.id ?
                                  "Commit " + root.snapshot.commitShort : "Snapshot details"
                            color: Theme.text
                            font.pixelSize: 18
                            font.weight: Font.DemiBold
                        }

                        GridLayout {
                            Layout.fillWidth: true
                            columns: 2
                            columnSpacing: 12
                            rowSpacing: 5
                            visible: !!root.snapshot.id

                            Text { text: "Adapter"; color: Theme.dim; font.pixelSize: 10 }
                            Text {
                                Layout.fillWidth: true
                                text: (root.snapshot.adapterId || "") + " v" +
                                      (root.snapshot.adapterVersion || "")
                                color: Theme.text2
                                font.pixelSize: 10
                                wrapMode: Text.WrapAnywhere
                            }

                            Text { text: "Contract"; color: Theme.dim; font.pixelSize: 10 }
                            Text {
                                Layout.fillWidth: true
                                text: root.snapshot.contractVersion || ""
                                color: Theme.text2
                                font.pixelSize: 10
                            }

                            Text { text: "Commit SHA"; color: Theme.dim; font.pixelSize: 10 }
                            Text {
                                Layout.fillWidth: true
                                text: root.snapshot.commitSha || ""
                                color: Theme.text2
                                font.pixelSize: 10
                                wrapMode: Text.WrapAnywhere
                            }

                            Text { text: "Repository"; color: Theme.dim; font.pixelSize: 10 }
                            Text {
                                Layout.fillWidth: true
                                text: root.snapshot.repositoryIdentity || ""
                                color: Theme.text2
                                font.pixelSize: 10
                                wrapMode: Text.WrapAnywhere
                            }

                            Text { text: "Local path"; color: Theme.dim; font.pixelSize: 10 }
                            Text {
                                Layout.fillWidth: true
                                text: root.snapshot.repositoryPath || ""
                                color: Theme.text2
                                font.pixelSize: 10
                                wrapMode: Text.WrapAnywhere
                            }

                            Text { text: "Dirty at capture"; color: Theme.dim; font.pixelSize: 10 }
                            StatusPill {
                                text: root.snapshot.workingTreeDirty ? "YES · EXCLUDED" : "NO"
                                tone: root.snapshot.workingTreeDirty ? "warn" : "good"
                            }

                            Text { text: "Compatibility"; color: Theme.dim; font.pixelSize: 10 }
                            Text {
                                Layout.fillWidth: true
                                text: root.snapshot.compatibilitySignature || ""
                                color: Theme.text2
                                font.pixelSize: 10
                                wrapMode: Text.WrapAnywhere
                            }

                            Text { text: "Manifest SHA"; color: Theme.dim; font.pixelSize: 10 }
                            Text {
                                Layout.fillWidth: true
                                text: root.snapshot.manifestSha256 || ""
                                color: Theme.text2
                                font.pixelSize: 10
                                wrapMode: Text.WrapAnywhere
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                            color: Theme.border
                            visible: !!root.snapshot.id
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            visible: !!root.snapshot.id

                            Text {
                                Layout.fillWidth: true
                                text: "DECLARED FILES"
                                color: Theme.dim
                                font.pixelSize: 10
                                font.weight: Font.Bold
                            }

                            Text {
                                text: String(root.snapshot.fileCount || 0)
                                color: Theme.muted
                                font.pixelSize: 10
                            }
                        }

                        Repeater {
                            model: contracts.selectedFiles

                            delegate: Rectangle {
                                required property var modelData
                                Layout.fillWidth: true
                                implicitHeight: fileColumn.implicitHeight + 14
                                radius: 6
                                color: Theme.surfaceAlt
                                border.color: Theme.border

                                ColumnLayout {
                                    id: fileColumn
                                    anchors.fill: parent
                                    anchors.margins: 7
                                    spacing: 2

                                    RowLayout {
                                        Layout.fillWidth: true

                                        Text {
                                            Layout.fillWidth: true
                                            text: modelData.path
                                            color: Theme.text
                                            font.pixelSize: 10
                                            font.weight: Font.DemiBold
                                            elide: Text.ElideMiddle
                                        }

                                        StatusPill {
                                            text: modelData.role
                                            tone: "neutral"
                                        }
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.sha256 + " · " +
                                              modelData.sizeBytes + " bytes" +
                                              (modelData.required ? " · required" : " · optional")
                                        color: Theme.dim
                                        font.pixelSize: 9
                                        wrapMode: Text.WrapAnywhere
                                    }
                                }
                            }
                        }

                        Text {
                            Layout.fillWidth: true
                            visible: !root.snapshot.id
                            text: "Select a frozen snapshot to inspect its commit, compatibility signature and per-file hashes."
                            color: Theme.dim
                            font.pixelSize: 11
                            wrapMode: Text.WordWrap
                        }
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true

            Text {
                Layout.fillWidth: true
                text: "Snapshots are Lab provenance only. They do not change Frankenhomie or authorize integration."
                color: Theme.dim
                font.pixelSize: 10
                wrapMode: Text.WordWrap
            }

            LabButton {
                text: "Close"
                onClicked: root.close()
            }
        }
    }

    onOpened: {
        if (repositoryPath.text.length === 0 && contracts.suggestedRepository.length > 0)
            repositoryPath.text = contracts.suggestedRepository
    }
}
