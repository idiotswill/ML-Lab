import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    id: root

    readonly property var lab: appController.experiments
    readonly property var project: appController.selectedProject
    readonly property var selected: lab.selectedExperiment
    readonly property var selectedJob: lab.selectedJob
    readonly property var metrics: lab.selectedMetrics
    readonly property bool trainerUsesPayloadKeys:
        trainerBox.currentIndex >= 0 &&
        trainerBox.currentIndex < lab.trainerOptions.length &&
        Boolean(lab.trainerOptions[trainerBox.currentIndex].usesPayloadKeys)

    function statusTone(status) {
        if (status === "COMPLETED")
            return "good"
        if (status === "FAILED" || status === "CANCELLED" || status === "INTERRUPTED")
            return "bad"
        if (status === "RUNNING" || status === "QUEUED")
            return "warn"
        return "neutral"
    }

    function trainerOption() {
        if (trainerBox.currentIndex < 0 || trainerBox.currentIndex >= lab.trainerOptions.length)
            return null
        return lab.trainerOptions[trainerBox.currentIndex]
    }

    function applyTrainerDefaults() {
        const option = trainerOption()
        if (!option)
            return
        featureDimField.text = String(option.defaultFeatureDim)
        alphaField.text = String(option.defaultAlpha)
        textKeyField.text = String(option.defaultTextKey)
        labelKeyField.text = String(option.defaultLabelKey)
        for (let i = 0; i < lab.runtimeOptions.length; ++i) {
            if (lab.runtimeOptions[i].id === option.runtimeId) {
                runtimeBox.currentIndex = i
                break
            }
        }
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
                    text: "Experiments & Training"
                    color: Theme.text
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }

                Text {
                    text: root.project.name ?
                              root.project.name + "  ·  " + root.project.adapter_id :
                              "Select a project to launch experiments"
                    color: Theme.muted
                    font.pixelSize: 12
                }
            }

            StatusPill {
                text: "TRAIN ONLY · TEST/REDTEAM SEALED"
                tone: "good"
            }
        }

        Panel {
            visible: !lab.hasProject
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.centerIn: parent
                width: Math.min(parent.width - 48, 560)
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
                    text: "Experiments are project-scoped. Open a project, freeze a dataset in Data Studio, then launch a packaged trainer here."
                    color: Theme.muted
                    font.pixelSize: 13
                    wrapMode: Text.WordWrap
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }

        RowLayout {
            visible: lab.hasProject
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 14

            Panel {
                Layout.preferredWidth: 306
                Layout.minimumWidth: 250
                Layout.fillHeight: true

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            text: "Experiment history"
                            color: Theme.text
                            font.pixelSize: 13
                            font.weight: Font.DemiBold
                        }

                        Item {
                            Layout.fillWidth: true
                        }

                        Text {
                            text: String(lab.experiments.length)
                            color: Theme.dim
                            font.pixelSize: 11
                        }
                    }

                    Text {
                        visible: lab.experiments.length === 0
                        Layout.fillWidth: true
                        text: "No experiments yet. Launch one from the configuration panel."
                        color: Theme.muted
                        font.pixelSize: 12
                        wrapMode: Text.WordWrap
                    }

                    ListView {
                        id: experimentList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 6
                        model: lab.experiments

                        delegate: Button {
                            id: experimentButton
                            required property var modelData
                            width: experimentList.width
                            height: 78
                            activeFocusOnTab: true
                            Accessible.role: Accessible.Button
                            Accessible.name: modelData.datasetName + ", " + modelData.status
                            onClicked: lab.selectExperiment(String(modelData.id))

                            contentItem: Column {
                                leftPadding: 10
                                rightPadding: 8
                                spacing: 4

                                Row {
                                    width: parent.width - 18
                                    spacing: 8

                                    Text {
                                        width: Math.max(40, parent.width - statusLabel.width - 12)
                                        text: experimentButton.modelData.datasetName
                                        color: Theme.text
                                        font.pixelSize: 13
                                        font.weight: Font.DemiBold
                                        elide: Text.ElideRight
                                    }

                                    Text {
                                        id: statusLabel
                                        text: experimentButton.modelData.status
                                        color: root.statusTone(experimentButton.modelData.status) === "good" ?
                                                   Theme.goodText :
                                               root.statusTone(experimentButton.modelData.status) === "bad" ?
                                                   Theme.badText : Theme.warnText
                                        font.pixelSize: 10
                                        font.weight: Font.Bold
                                    }
                                }

                                Text {
                                    width: parent.width - 18
                                    text: experimentButton.modelData.trainerId
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    elide: Text.ElideMiddle
                                }

                                Text {
                                    width: parent.width - 18
                                    text: experimentButton.modelData.id.substring(0, 8) +
                                          "  ·  seed " + experimentButton.modelData.seed
                                    color: Theme.dim
                                    font.pixelSize: 10
                                    elide: Text.ElideRight
                                }
                            }

                            background: Rectangle {
                                radius: 8
                                color: root.selected.id === experimentButton.modelData.id ?
                                           Theme.accentSurface :
                                           (experimentButton.hovered ? Theme.hover : Theme.surfaceAlt)
                                border.color: experimentButton.activeFocus ? Theme.accent :
                                              (root.selected.id === experimentButton.modelData.id ?
                                                   Theme.accentBorder : Theme.border)
                                border.width: experimentButton.activeFocus ? 2 : 1
                            }
                        }

                        ScrollBar.vertical: ScrollBar {}
                    }
                }
            }

            ScrollView {
                id: detailsScroll
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

                ColumnLayout {
                    width: detailsScroll.availableWidth
                    spacing: 14

                    Panel {
                        Layout.fillWidth: true
                        implicitHeight: launcherContent.implicitHeight + 32

                        ColumnLayout {
                            id: launcherContent
                            anchors.fill: parent
                            anchors.margins: 16
                            spacing: 12

                            RowLayout {
                                Layout.fillWidth: true

                                Text {
                                    text: "Launch experiment"
                                    color: Theme.text
                                    font.pixelSize: 16
                                    font.weight: Font.DemiBold
                                }

                                Item {
                                    Layout.fillWidth: true
                                }

                                StatusPill {
                                    visible: !lab.hasRunnableTrainer
                                    text: "NO RUNNABLE TRAINER"
                                    tone: "warn"
                                }
                            }

                            Text {
                                visible: !lab.hasRunnableTrainer
                                Layout.fillWidth: true
                                text: "This project adapter has no packaged training worker yet. Experimental model classes are not advertised as runnable until they pass through the isolated worker contract."
                                color: Theme.warnText
                                font.pixelSize: 12
                                wrapMode: Text.WordWrap
                            }

                            Text {
                                visible: lab.hasRunnableTrainer && lab.frozenDatasets.length === 0
                                Layout.fillWidth: true
                                text: "Freeze at least one dataset in Data Studio before training."
                                color: Theme.warnText
                                font.pixelSize: 12
                                wrapMode: Text.WordWrap
                            }

                            GridLayout {
                                visible: lab.hasRunnableTrainer
                                Layout.fillWidth: true
                                columns: 4
                                rowSpacing: 8
                                columnSpacing: 10

                                Text {
                                    text: "Dataset"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                ComboBox {
                                    id: datasetBox
                                    Layout.fillWidth: true
                                    model: lab.frozenDatasets
                                    textRole: "name"
                                    valueRole: "id"
                                    activeFocusOnTab: true
                                    Accessible.name: "Frozen dataset"
                                }

                                Text {
                                    text: "Seed"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                TextField {
                                    id: seedField
                                    Layout.preferredWidth: 110
                                    text: "0"
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    activeFocusOnTab: true
                                    Accessible.name: "Training seed"
                                }

                                Text {
                                    text: "Trainer"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                ComboBox {
                                    id: trainerBox
                                    Layout.fillWidth: true
                                    model: lab.trainerOptions
                                    textRole: "name"
                                    valueRole: "trainerId"
                                    activeFocusOnTab: true
                                    Accessible.name: "Trainer"
                                    onCurrentIndexChanged: root.applyTrainerDefaults()
                                    Component.onCompleted: root.applyTrainerDefaults()
                                }

                                Text {
                                    text: "Runtime"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                ComboBox {
                                    id: runtimeBox
                                    Layout.fillWidth: true
                                    model: lab.runtimeOptions
                                    textRole: "name"
                                    valueRole: "id"
                                    activeFocusOnTab: true
                                    Accessible.name: "Runtime pack"
                                }

                                Text {
                                    text: "Feature dim"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                TextField {
                                    id: featureDimField
                                    Layout.fillWidth: true
                                    text: "32768"
                                    inputMethodHints: Qt.ImhDigitsOnly
                                    activeFocusOnTab: true
                                    Accessible.name: "Feature dimension"
                                }

                                Text {
                                    text: "Alpha"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                TextField {
                                    id: alphaField
                                    Layout.fillWidth: true
                                    text: "0.5"
                                    inputMethodHints: Qt.ImhFormattedNumbersOnly
                                    activeFocusOnTab: true
                                    Accessible.name: "Smoothing alpha"
                                }

                                Text {
                                    visible: root.trainerUsesPayloadKeys
                                    text: "Payload key"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                TextField {
                                    id: textKeyField
                                    visible: root.trainerUsesPayloadKeys
                                    Layout.fillWidth: true
                                    text: "text"
                                    activeFocusOnTab: true
                                    Accessible.name: "Payload text key"
                                }

                                Text {
                                    visible: root.trainerUsesPayloadKeys
                                    text: "Label key"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    font.weight: Font.Bold
                                }

                                TextField {
                                    id: labelKeyField
                                    visible: root.trainerUsesPayloadKeys
                                    Layout.fillWidth: true
                                    text: "class"
                                    activeFocusOnTab: true
                                    Accessible.name: "Label key"
                                }
                            }

                            Text {
                                visible: lab.hasRunnableTrainer && !root.trainerUsesPayloadKeys
                                Layout.fillWidth: true
                                text: "This trainer consumes the adapter-validated bounded request/label contract directly; freeform payload and label keys are not configurable."
                                color: Theme.dim
                                font.pixelSize: 10
                                wrapMode: Text.WordWrap
                            }

                            RowLayout {
                                visible: lab.hasRunnableTrainer
                                Layout.fillWidth: true

                                Text {
                                    Layout.fillWidth: true
                                    text: "Worker receives TRAIN + optional DEV artifacts only. TEST and REDTEAM paths are withheld by construction."
                                    color: Theme.dim
                                    font.pixelSize: 10
                                    wrapMode: Text.WordWrap
                                }

                                LabButton {
                                    text: "Create & launch"
                                    primary: true
                                    enabled: lab.frozenDatasets.length > 0 &&
                                             trainerBox.currentIndex >= 0 &&
                                             runtimeBox.currentIndex >= 0
                                    onClicked: lab.createAndLaunch(
                                        String(datasetBox.currentValue),
                                        String(trainerBox.currentValue),
                                        String(runtimeBox.currentValue),
                                        seedField.text,
                                        featureDimField.text,
                                        alphaField.text,
                                        textKeyField.text,
                                        labelKeyField.text
                                    )
                                }
                            }
                        }
                    }

                    Panel {
                        Layout.fillWidth: true
                        implicitHeight: Math.max(280, experimentDetails.implicitHeight + 32)

                        ColumnLayout {
                            id: experimentDetails
                            anchors.fill: parent
                            anchors.margins: 16
                            spacing: 12

                            ColumnLayout {
                                visible: !root.selected.id
                                Layout.fillWidth: true
                                Layout.preferredHeight: 220
                                spacing: 8

                                Item {
                                    Layout.fillHeight: true
                                }

                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "Select an experiment"
                                    color: Theme.text
                                    font.pixelSize: 18
                                    font.weight: Font.DemiBold
                                }

                                Text {
                                    Layout.alignment: Qt.AlignHCenter
                                    text: "Progress, metrics and immutable artifact hashes will appear here."
                                    color: Theme.muted
                                    font.pixelSize: 12
                                }

                                Item {
                                    Layout.fillHeight: true
                                }
                            }

                            ColumnLayout {
                                visible: !!root.selected.id
                                Layout.fillWidth: true
                                spacing: 12

                                RowLayout {
                                    Layout.fillWidth: true

                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 2

                                        RowLayout {
                                            Text {
                                                text: root.selected.datasetName || ""
                                                color: Theme.text
                                                font.pixelSize: 19
                                                font.weight: Font.DemiBold
                                            }

                                            StatusPill {
                                                text: root.selected.status || ""
                                                tone: root.statusTone(root.selected.status || "")
                                            }
                                        }

                                        Text {
                                            text: (root.selected.trainerId || "") + "  ·  " +
                                                  (root.selected.runtimeId || "") + "  ·  seed " +
                                                  String(root.selected.seed || 0)
                                            color: Theme.muted
                                            font.pixelSize: 11
                                        }
                                    }

                                    LabButton {
                                        visible: lab.canCancelSelected
                                        text: "Cancel training"
                                        onClicked: lab.cancelSelected()
                                    }
                                }

                                ColumnLayout {
                                    visible: root.selected.status === "RUNNING" ||
                                             root.selectedJob.status === "CANCELLING"
                                    Layout.fillWidth: true
                                    spacing: 5

                                    ProgressBar {
                                        Layout.fillWidth: true
                                        from: 0
                                        to: 1
                                        value: Number(root.selectedJob.progress || 0)
                                        Accessible.name: "Training progress"
                                    }

                                    RowLayout {
                                        Layout.fillWidth: true

                                        Text {
                                            Layout.fillWidth: true
                                            text: root.selectedJob.message || "Running"
                                            color: Theme.muted
                                            font.pixelSize: 11
                                        }

                                        Text {
                                            text: Math.round(Number(root.selectedJob.progress || 0) * 100) + "%"
                                            color: Theme.text2
                                            font.pixelSize: 11
                                            font.weight: Font.DemiBold
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    height: 1
                                    color: Theme.border
                                }

                                GridLayout {
                                    Layout.fillWidth: true
                                    columns: 2
                                    rowSpacing: 7
                                    columnSpacing: 14

                                    Text {
                                        text: "Experiment"
                                        color: Theme.dim
                                        font.pixelSize: 10
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: root.selected.id || ""
                                        color: Theme.text2
                                        font.pixelSize: 10
                                        elide: Text.ElideMiddle
                                    }

                                    Text {
                                        text: "Created"
                                        color: Theme.dim
                                        font.pixelSize: 10
                                    }

                                    Text {
                                        text: root.selected.createdAt || ""
                                        color: Theme.text2
                                        font.pixelSize: 10
                                    }

                                    Text {
                                        text: "Model SHA-256"
                                        color: Theme.dim
                                        font.pixelSize: 10
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: root.selected.modelDigest || "—"
                                        color: root.selected.modelDigest ? Theme.goodText : Theme.dim
                                        font.pixelSize: 10
                                        elide: Text.ElideMiddle
                                    }

                                    Text {
                                        text: "Manifest SHA-256"
                                        color: Theme.dim
                                        font.pixelSize: 10
                                    }

                                    Text {
                                        Layout.fillWidth: true
                                        text: root.selected.manifestDigest || "—"
                                        color: root.selected.manifestDigest ? Theme.goodText : Theme.dim
                                        font.pixelSize: 10
                                        elide: Text.ElideMiddle
                                    }
                                }

                                Text {
                                    text: "Metrics"
                                    color: Theme.text
                                    font.pixelSize: 13
                                    font.weight: Font.DemiBold
                                }

                                Text {
                                    visible: root.metrics.length === 0
                                    text: root.selected.status === "COMPLETED" ?
                                              "No metrics were emitted by this trainer." :
                                              "Metrics are committed only after successful completion."
                                    color: Theme.muted
                                    font.pixelSize: 11
                                }

                                Flow {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Repeater {
                                        model: root.metrics

                                        delegate: Rectangle {
                                            required property var modelData
                                            width: metricText.implicitWidth + 20
                                            height: 30
                                            radius: 7
                                            color: modelData.veto ?
                                                       Theme.warnSurface : Theme.surfaceAlt
                                            border.color: modelData.veto ?
                                                              Theme.warnBorder : Theme.border

                                            Text {
                                                id: metricText
                                                anchors.centerIn: parent
                                                text: modelData.metricId + "  " +
                                                      Number(modelData.value).toFixed(4)
                                                color: modelData.veto ?
                                                           Theme.warnText : Theme.text2
                                                font.pixelSize: 10
                                                font.weight: Font.DemiBold
                                            }
                                        }
                                    }
                                }

                                Text {
                                    visible: !!root.selectedJob.error
                                    Layout.fillWidth: true
                                    text: root.selectedJob.error || ""
                                    color: Theme.badText
                                    font.pixelSize: 11
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
