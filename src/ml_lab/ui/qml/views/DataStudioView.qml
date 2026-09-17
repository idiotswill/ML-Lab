import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import "../components"

Item {
    id: root

    readonly property var studio: appController.dataStudio
    readonly property var project: appController.selectedProject
    readonly property var selectedDataset: studio.selectedDataset
    readonly property var counts: studio.splitCounts

    FileDialog {
        id: importDialog
        title: "Import JSON Lines dataset"
        fileMode: FileDialog.OpenFile
        nameFilters: ["JSON Lines (*.jsonl)", "All files (*)"]
        onAccepted: studio.importJsonl(selectedFile)
    }

    Dialog {
        id: createDialog
        modal: true
        anchors.centerIn: parent
        width: 420
        title: "Create dataset"
        standardButtons: Dialog.NoButton

        background: Rectangle {
            color: Theme.surface
            radius: 12
            border.color: Theme.border
        }

        contentItem: ColumnLayout {
            spacing: 14

            Text {
                text: "Dataset name"
                color: Theme.text
                font.pixelSize: 13
                font.weight: Font.DemiBold
            }

            TextField {
                id: datasetName
                Layout.fillWidth: true
                placeholderText: "e.g. Residual semantics September"
                activeFocusOnTab: true
                selectByMouse: true
                onAccepted: {
                    if (text.trim().length > 0) {
                        studio.createDataset(text)
                        createDialog.close()
                        text = ""
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true

                Item { Layout.fillWidth: true }

                LabButton {
                    text: "Cancel"
                    onClicked: createDialog.close()
                }

                LabButton {
                    text: "Create"
                    primary: true
                    enabled: datasetName.text.trim().length > 0
                    onClicked: {
                        studio.createDataset(datasetName.text)
                        createDialog.close()
                        datasetName.text = ""
                    }
                }
            }
        }

        onOpened: datasetName.forceActiveFocus()
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 3

                Text {
                    text: "Data Studio"
                    color: Theme.text
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }

                Text {
                    text: root.project.name ?
                              root.project.name + "  ·  " + root.project.adapter_id :
                              "Select a project to manage datasets"
                    color: Theme.muted
                    font.pixelSize: 12
                }
            }

            LabButton {
                text: "New dataset"
                primary: true
                enabled: studio.hasProject && !studio.busy
                onClicked: createDialog.open()
            }
        }

        Panel {
            visible: !studio.hasProject
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
                    text: "Data Studio is project-scoped. Open a project from Projects, then return here to import, inspect, leak-check and freeze its dataset versions."
                    color: Theme.muted
                    font.pixelSize: 13
                    wrapMode: Text.WordWrap
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }

        RowLayout {
            visible: studio.hasProject
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 14

            Panel {
                Layout.preferredWidth: 272
                Layout.minimumWidth: 230
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 10

                    Text {
                        text: "Dataset versions"
                        color: Theme.text
                        font.pixelSize: 13
                        font.weight: Font.DemiBold
                    }

                    Text {
                        visible: studio.datasets.length === 0
                        Layout.fillWidth: true
                        text: "No datasets yet. Create a draft version to start importing examples."
                        color: Theme.muted
                        font.pixelSize: 12
                        wrapMode: Text.WordWrap
                    }

                    ListView {
                        id: datasetList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 6
                        model: studio.datasets

                        delegate: Button {
                            id: datasetButton
                            required property var modelData
                            width: datasetList.width
                            height: 64
                            activeFocusOnTab: true
                            Accessible.role: Accessible.Button
                            Accessible.name: modelData.name + ", " + modelData.state
                            onClicked: studio.selectDataset(String(modelData.id))

                            contentItem: Column {
                                leftPadding: 10
                                rightPadding: 8
                                spacing: 4

                                Text {
                                    width: parent.width - 18
                                    text: datasetButton.modelData.name
                                    color: Theme.text
                                    font.pixelSize: 13
                                    font.weight: Font.DemiBold
                                    elide: Text.ElideRight
                                }

                                Text {
                                    width: parent.width - 18
                                    text: datasetButton.modelData.state + "  ·  " +
                                          datasetButton.modelData.example_count + " examples"
                                    color: datasetButton.modelData.state === "FROZEN" ?
                                               Theme.goodText : Theme.muted
                                    font.pixelSize: 11
                                    elide: Text.ElideRight
                                }
                            }

                            background: Rectangle {
                                radius: 8
                                color: root.selectedDataset.id === datasetButton.modelData.id ?
                                           Theme.accentSurface :
                                           (datasetButton.hovered ? Theme.hover : Theme.surfaceAlt)
                                border.color: datasetButton.activeFocus ? Theme.accent :
                                              (root.selectedDataset.id === datasetButton.modelData.id ?
                                                   Theme.accentBorder : Theme.border)
                                border.width: datasetButton.activeFocus ? 2 : 1
                            }
                        }
                    }
                }
            }

            Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 12

                    ColumnLayout {
                        visible: !root.selectedDataset.id
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 8

                        Item { Layout.fillHeight: true }

                        Text {
                            Layout.alignment: Qt.AlignHCenter
                            text: "Select a dataset version"
                            color: Theme.text
                            font.pixelSize: 18
                            font.weight: Font.DemiBold
                        }

                        Text {
                            Layout.alignment: Qt.AlignHCenter
                            text: "Examples are paged from SQLite; large datasets are never materialized into QML at once."
                            color: Theme.muted
                            font.pixelSize: 12
                        }

                        Item { Layout.fillHeight: true }
                    }

                    ColumnLayout {
                        visible: !!root.selectedDataset.id
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 12

                        RowLayout {
                            Layout.fillWidth: true

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2

                                RowLayout {
                                    Text {
                                        text: root.selectedDataset.name || ""
                                        color: Theme.text
                                        font.pixelSize: 19
                                        font.weight: Font.DemiBold
                                    }

                                    StatusPill {
                                        text: root.selectedDataset.state || ""
                                        tone: root.selectedDataset.state === "FROZEN" ? "good" : "neutral"
                                    }
                                }

                                Text {
                                    text: root.selectedDataset.state === "FROZEN" ?
                                              "Immutable split artifacts · training sees TRAIN/DEV only" :
                                              "Draft dataset · import and validate before freeze"
                                    color: Theme.muted
                                    font.pixelSize: 11
                                }
                            }

                            LabButton {
                                text: "Import JSONL"
                                enabled: root.selectedDataset.state === "DRAFT" && !studio.busy
                                onClicked: importDialog.open()
                            }

                            LabButton {
                                text: "Freeze"
                                primary: true
                                enabled: root.selectedDataset.state === "DRAFT" &&
                                         root.selectedDataset.example_count > 0 && !studio.busy
                                onClicked: studio.freezeSelectedDataset()
                            }
                        }

                        ProgressBar {
                            visible: studio.busy
                            Layout.fillWidth: true
                            indeterminate: true
                            Accessible.name: studio.busyMessage
                        }

                        Text {
                            visible: studio.busy
                            text: studio.busyMessage
                            color: Theme.muted
                            font.pixelSize: 11
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Repeater {
                                model: ["ALL", "TRAIN", "DEV", "TEST", "REDTEAM"]

                                delegate: Button {
                                    required property string modelData
                                    checkable: true
                                    checked: studio.splitFilter === modelData
                                    activeFocusOnTab: true
                                    text: modelData + "  " + Number(root.counts[modelData] || 0)
                                    implicitHeight: 32
                                    onClicked: studio.setSplitFilter(modelData)

                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.checked ? Theme.goodText : Theme.text2
                                        font.pixelSize: 11
                                        font.weight: parent.checked ? Font.DemiBold : Font.Normal
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }

                                    background: Rectangle {
                                        radius: 7
                                        color: parent.checked ? Theme.goodSurface : Theme.surfaceAlt
                                        border.color: parent.activeFocus ? Theme.accent :
                                                      (parent.checked ? Theme.goodBorder : Theme.border)
                                        border.width: parent.activeFocus ? 2 : 1
                                    }
                                }
                            }

                            Item { Layout.fillWidth: true }

                            Text {
                                text: "Page " + studio.pageNumber
                                color: Theme.dim
                                font.pixelSize: 11
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            implicitHeight: 34
                            color: Theme.sidebar
                            radius: 6

                            RowLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 10
                                anchors.rightMargin: 10
                                spacing: 10

                                Text {
                                    Layout.preferredWidth: 150
                                    text: "Example"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }
                                Text {
                                    Layout.preferredWidth: 76
                                    text: "Split"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }
                                Text {
                                    Layout.preferredWidth: 170
                                    text: "Source / lineage"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: "Payload"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }
                                Text {
                                    Layout.preferredWidth: 180
                                    text: "Expected"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }
                            }
                        }

                        ListView {
                            id: exampleList
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            clip: true
                            spacing: 1
                            model: studio.examples

                            delegate: Rectangle {
                                required property var modelData
                                width: exampleList.width
                                height: 48
                                color: index % 2 === 0 ? Theme.surfaceAlt : Theme.surface
                                radius: 4

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 10
                                    anchors.rightMargin: 10
                                    spacing: 10

                                    Text {
                                        Layout.preferredWidth: 150
                                        text: modelData.exampleId
                                        color: Theme.text2
                                        font.pixelSize: 11
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        Layout.preferredWidth: 76
                                        text: modelData.split
                                        color: modelData.split === "TEST" || modelData.split === "REDTEAM" ?
                                                   Theme.warnText : Theme.muted
                                        font.pixelSize: 10
                                        font.weight: Font.DemiBold
                                    }
                                    Text {
                                        Layout.preferredWidth: 170
                                        text: modelData.sourceId + " / " + modelData.lineageGroup
                                        color: Theme.dim
                                        font.pixelSize: 10
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.payload
                                        color: Theme.text2
                                        font.pixelSize: 10
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        Layout.preferredWidth: 180
                                        text: modelData.label
                                        color: Theme.text2
                                        font.pixelSize: 10
                                        elide: Text.ElideRight
                                    }
                                }
                            }

                            ScrollBar.vertical: ScrollBar {}
                        }

                        Text {
                            visible: studio.examples.length === 0
                            Layout.alignment: Qt.AlignHCenter
                            text: root.selectedDataset.example_count === 0 ?
                                      "No examples imported yet." :
                                      "No examples in this split."
                            color: Theme.muted
                            font.pixelSize: 12
                        }

                        RowLayout {
                            Layout.fillWidth: true

                            LabButton {
                                text: "Previous"
                                enabled: studio.canPreviousPage
                                onClicked: studio.previousPage()
                            }

                            LabButton {
                                text: "Next"
                                enabled: studio.canNextPage
                                onClicked: studio.nextPage()
                            }

                            Item { Layout.fillWidth: true }

                            Text {
                                text: Number(root.counts[studio.splitFilter] || 0) +
                                      " visible examples · 100 per page"
                                color: Theme.dim
                                font.pixelSize: 10
                            }
                        }
                    }
                }
            }
        }
    }
}
