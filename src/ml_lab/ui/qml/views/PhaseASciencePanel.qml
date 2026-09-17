import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Panel {
    id: root

    required property var compareController
    required property var scienceController

    implicitHeight: 232

    function providerMetric(metricId) {
        const metrics = scienceController.providerReference.metrics || []
        for (let i = 0; i < metrics.length; ++i) {
            if (metrics[i].metricId === metricId)
                return metrics[i]
        }
        return null
    }

    function metricText(metric) {
        if (!metric || !metric.measured)
            return "NOT MEASURED"
        if (metric.percent)
            return (Number(metric.value) * 100).toFixed(1) + "%"
        if (metric.unit === "ms")
            return Number(metric.value).toFixed(2) + " ms"
        if (metric.unit === "MB")
            return Number(metric.value).toFixed(2) + " MB"
        return Number(metric.value).toFixed(4)
    }

    Connections {
        target: root.compareController

        function onChanged() {
            root.scienceController.setSelection(
                String(root.compareController.selectedExperiment.id || ""),
                root.compareController.split
            )
        }

        function onOperationCompleted(_message) {
            root.scienceController.refresh()
        }
    }

    Connections {
        target: root.scienceController

        function onBaselineReady(experimentId) {
            root.compareController.selectExperiment(experimentId)
        }
    }

    Component.onCompleted: scienceController.setSelection(
        String(compareController.selectedExperiment.id || ""),
        compareController.split
    )

    RowLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 12

        ColumnLayout {
            Layout.preferredWidth: 230
            Layout.minimumWidth: 205
            Layout.fillHeight: true
            spacing: 7

            RowLayout {
                Layout.fillWidth: true

                Text {
                    Layout.fillWidth: true
                    text: "PHASE A SCIENCE"
                    color: Theme.text
                    font.pixelSize: 11
                    font.weight: Font.Bold
                }

                StatusPill {
                    text: "VETO-FIRST"
                    tone: "warn"
                }
            }

            Text {
                Layout.fillWidth: true
                text: "Rerun baselines on this experiment's same frozen dataset + contract."
                color: Theme.muted
                font.pixelSize: 10
                wrapMode: Text.WordWrap
            }

            Repeater {
                model: scienceController.baselineOptions

                delegate: RowLayout {
                    required property var modelData
                    Layout.fillWidth: true
                    spacing: 6

                    LabButton {
                        Layout.fillWidth: true
                        text: modelData.completed ? modelData.name + " ✓" : modelData.name
                        enabled: !scienceController.busy
                        onClicked: scienceController.runBaseline(modelData.baselineId)
                    }
                }
            }

            Text {
                visible: scienceController.baselineOptions.length === 0
                Layout.fillWidth: true
                text: "Select a completed Phase A experiment with a pinned contract."
                color: Theme.dim
                font.pixelSize: 10
                wrapMode: Text.WordWrap
            }

            RowLayout {
                Layout.fillWidth: true
                visible: scienceController.busy

                BusyIndicator {
                    running: visible
                    implicitWidth: 18
                    implicitHeight: 18
                }

                Text {
                    Layout.fillWidth: true
                    text: scienceController.busyMessage
                    color: Theme.muted
                    font.pixelSize: 10
                    elide: Text.ElideRight
                }

                LabButton {
                    text: "Cancel"
                    onClicked: scienceController.cancelBaseline()
                }
            }

            Item { Layout.fillHeight: true }

            Text {
                Layout.fillWidth: true
                text: "Historical provider numbers are reference evidence only; they are not a same-dataset ranking."
                color: Theme.warnText
                font.pixelSize: 9
                wrapMode: Text.WordWrap
            }
        }

        Rectangle {
            Layout.fillHeight: true
            width: 1
            color: Theme.border
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 6

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Text {
                    Layout.preferredWidth: 190
                    text: "METRIC"
                    color: Theme.dim
                    font.pixelSize: 9
                    font.weight: Font.Bold
                }

                Text {
                    Layout.fillWidth: true
                    text: "SELECTED · " + compareController.split
                    color: Theme.text2
                    font.pixelSize: 9
                    font.weight: Font.Bold
                }

                ColumnLayout {
                    Layout.preferredWidth: 220
                    spacing: 0

                    Text {
                        Layout.fillWidth: true
                        text: scienceController.providerReference.available ?
                                  "QWEN HISTORICAL · DIFFERENT CORPUS" :
                                  "HISTORICAL PROVIDER"
                        color: Theme.warnText
                        font.pixelSize: 9
                        font.weight: Font.Bold
                        elide: Text.ElideRight
                    }

                    Text {
                        Layout.fillWidth: true
                        text: scienceController.providerReference.available ?
                                  String(scienceController.providerReference.model) +
                                  " · " + String(scienceController.providerReference.modelCalls) +
                                  " calls" : "No pinned provider result"
                        color: Theme.dim
                        font.pixelSize: 8
                        elide: Text.ElideRight
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: Theme.border
            }

            ListView {
                id: metricList
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                spacing: 1
                model: scienceController.scientificMetrics

                delegate: Rectangle {
                    required property var modelData
                    width: metricList.width
                    height: 28
                    radius: 4
                    color: modelData.veto ? Theme.warnSurface :
                           (index % 2 === 0 ? Theme.surfaceAlt : "transparent")

                    readonly property var historical: root.providerMetric(modelData.metricId)

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 6
                        anchors.rightMargin: 6
                        spacing: 8

                        Text {
                            Layout.preferredWidth: 184
                            text: modelData.veto ? "VETO · " + modelData.label : modelData.label
                            color: modelData.veto ? Theme.warnText : Theme.text2
                            font.pixelSize: 9
                            font.weight: modelData.veto ? Font.Bold : Font.Normal
                            elide: Text.ElideRight
                            ToolTip.visible: metricMouse.containsMouse
                            ToolTip.text: modelData.note

                            MouseArea {
                                id: metricMouse
                                anchors.fill: parent
                                hoverEnabled: true
                                acceptedButtons: Qt.NoButton
                            }
                        }

                        Text {
                            Layout.fillWidth: true
                            text: root.metricText(modelData)
                            color: modelData.veto && modelData.measured &&
                                   Number(modelData.value) !== 0 ? Theme.badText : Theme.text
                            font.pixelSize: 10
                            font.weight: Font.DemiBold
                        }

                        Text {
                            Layout.preferredWidth: 220
                            text: root.metricText(parent.parent.historical)
                            color: parent.parent.historical &&
                                   parent.parent.historical.veto &&
                                   parent.parent.historical.measured &&
                                   Number(parent.parent.historical.value) !== 0 ?
                                       Theme.badText : Theme.muted
                            font.pixelSize: 10
                            font.weight: Font.DemiBold
                            elide: Text.ElideRight
                        }
                    }
                }

                ScrollBar.vertical: ScrollBar {}

                BusyIndicator {
                    anchors.centerIn: parent
                    visible: scienceController.loadingMetrics
                    running: visible
                }

                Text {
                    anchors.centerIn: parent
                    visible: !scienceController.loadingMetrics && metricList.count === 0
                    text: "Select a completed Phase A experiment."
                    color: Theme.dim
                    font.pixelSize: 11
                }
            }

            RowLayout {
                visible: scienceController.providerReference.available
                Layout.fillWidth: true
                spacing: 8

                Text {
                    Layout.fillWidth: true
                    text: "Pinned source " +
                          String(scienceController.providerReference.sourceHead).substring(0, 10) +
                          (scienceController.providerReference.sourceDirty ? " · source was dirty" : "") +
                          (scienceController.providerReference.independentQa ?
                               " · independent QA" : " · independent QA: NO")
                    color: Theme.dim
                    font.pixelSize: 8
                    elide: Text.ElideRight
                }
            }
        }
    }
}
