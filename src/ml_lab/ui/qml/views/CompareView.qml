import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    id: root

    ColumnLayout {
        anchors.fill: parent
        spacing: 12

        RowLayout {
            Layout.fillWidth: true

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2

                Text {
                    text: "Compare"
                    color: Theme.text
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }

                Text {
                    text: "Protected evaluation evidence · immutable cases · failure drilldown"
                    color: Theme.muted
                    font.pixelSize: 12
                }
            }

            LabButton {
                text: "TEST"
                checkable: true
                checked: appController.compare.split === "TEST"
                enabled: !appController.compare.busy
                onClicked: appController.compare.setSplit("TEST")
            }

            LabButton {
                text: "REDTEAM"
                checkable: true
                checked: appController.compare.split === "REDTEAM"
                enabled: !appController.compare.busy
                onClicked: appController.compare.setSplit("REDTEAM")
            }

            LabButton {
                text: appController.compare.busy ? "Evaluating…" : "Run evaluation"
                primary: true
                enabled: appController.compare.canRunEvaluation
                onClicked: appController.compare.runSelectedEvaluation()
            }

            LabButton {
                text: "Cancel"
                visible: appController.compare.busy
                enabled: appController.compare.busy
                onClicked: appController.compare.cancelEvaluation()
            }
        }

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: messageText.implicitHeight + 20
            radius: 8
            color: Theme.surfaceAlt
            border.color: Theme.border

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 10

                BusyIndicator {
                    visible: appController.compare.busy
                    running: visible
                    implicitWidth: 20
                    implicitHeight: 20
                }

                Text {
                    id: messageText
                    Layout.fillWidth: true
                    text: appController.compare.evaluatorMessage
                    color: Theme.muted
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                }
            }
        }

        Panel {
            Layout.fillWidth: true
            implicitHeight: 78

            RowLayout {
                anchors.fill: parent
                anchors.margins: 12
                spacing: 22

                ColumnLayout {
                    spacing: 2
                    Text { text: "EVIDENCE"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                    Text {
                        text: (appController.compare.selectedSummary.evaluated || 0) + " / " +
                              (appController.compare.selectedSummary.expected || 0)
                        color: Theme.text
                        font.pixelSize: 18
                        font.weight: Font.DemiBold
                    }
                }

                ColumnLayout {
                    spacing: 2
                    Text { text: "ACCURACY"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                    Text {
                        text: appController.compare.selectedSummary.evaluated ?
                              (Number(appController.compare.selectedSummary.accuracy) * 100).toFixed(1) + "%" : "—"
                        color: Theme.text
                        font.pixelSize: 18
                        font.weight: Font.DemiBold
                    }
                }

                ColumnLayout {
                    spacing: 2
                    Text { text: "FAILURES"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                    Text {
                        text: appController.compare.selectedSummary.failures || 0
                        color: Theme.text
                        font.pixelSize: 18
                        font.weight: Font.DemiBold
                    }
                }

                ColumnLayout {
                    spacing: 2
                    Text { text: "VETO"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                    Text {
                        text: appController.compare.selectedSummary.vetoFailures || 0
                        color: (appController.compare.selectedSummary.vetoFailures || 0) > 0 ? Theme.badText : Theme.text
                        font.pixelSize: 18
                        font.weight: Font.DemiBold
                    }
                }

                ColumnLayout {
                    spacing: 2
                    Text { text: "MEAN"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                    Text {
                        text: appController.compare.selectedSummary.evaluated ?
                              Number(appController.compare.selectedSummary.meanLatencyMs).toFixed(2) + " ms" : "—"
                        color: Theme.text
                        font.pixelSize: 18
                        font.weight: Font.DemiBold
                    }
                }

                ColumnLayout {
                    spacing: 2
                    Text { text: "P95"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                    Text {
                        text: appController.compare.selectedSummary.evaluated ?
                              Number(appController.compare.selectedSummary.p95LatencyMs).toFixed(2) + " ms" : "—"
                        color: Theme.text
                        font.pixelSize: 18
                        font.weight: Font.DemiBold
                    }
                }

                Item { Layout.fillWidth: true }

                StatusPill {
                    text: appController.compare.selectedSummary.complete ? "COMPLETE" :
                          (appController.compare.selectedSummary.evaluated || 0) > 0 ? "PARTIAL" : "NO EVIDENCE"
                    tone: appController.compare.selectedSummary.complete ? "good" :
                          (appController.compare.selectedSummary.evaluated || 0) > 0 ? "warn" : "neutral"
                }
            }
        }

        PhaseASciencePanel {
            visible: appController.compare.science.enabled
            Layout.fillWidth: true
            Layout.preferredHeight: visible ? implicitHeight : 0
            compareController: appController.compare
            scienceController: appController.compare.science
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
                            text: "COMPLETED EXPERIMENTS"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: appController.compare.experimentTotal + " total"
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                    }

                    ListView {
                        id: experimentList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 5
                        model: appController.compare.comparisonRows

                        delegate: ItemDelegate {
                            required property var modelData
                            width: experimentList.width
                            height: 78
                            activeFocusOnTab: true
                            Accessible.name: "Experiment " + modelData.shortId
                            onClicked: appController.compare.selectExperiment(modelData.id)

                            contentItem: ColumnLayout {
                                spacing: 3
                                RowLayout {
                                    Layout.fillWidth: true
                                    Text {
                                        Layout.fillWidth: true
                                        text: modelData.shortId + " · " + modelData.datasetName
                                        color: Theme.text
                                        font.pixelSize: 12
                                        font.weight: Font.DemiBold
                                        elide: Text.ElideRight
                                    }
                                    StatusPill {
                                        text: modelData.complete ? "DONE" :
                                              modelData.evaluated > 0 ? "PARTIAL" : "NONE"
                                        tone: modelData.complete ? "good" :
                                              modelData.evaluated > 0 ? "warn" : "neutral"
                                    }
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.trainerId
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    elide: Text.ElideMiddle
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.evaluated + "/" + modelData.expected +
                                          (modelData.evaluated > 0 ?
                                           " · " + (Number(modelData.accuracy) * 100).toFixed(1) + "%" : "") +
                                          " · veto " + modelData.vetoFailures
                                    color: modelData.vetoFailures > 0 ? Theme.badText : Theme.dim
                                    font.pixelSize: 10
                                    elide: Text.ElideRight
                                }
                            }

                            background: Rectangle {
                                radius: 7
                                color: appController.compare.selectedExperimentId === modelData.id ?
                                       Theme.accentSurface : (parent.hovered ? Theme.hover : "transparent")
                                border.color: parent.activeFocus ? Theme.accent :
                                              (appController.compare.selectedExperimentId === modelData.id ?
                                               Theme.accentBorder : "transparent")
                                border.width: parent.activeFocus ? 2 : 1
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: experimentList.count === 0
                            text: "No completed experiments yet."
                            color: Theme.dim
                            font.pixelSize: 12
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        LabButton {
                            text: "Previous"
                            enabled: appController.compare.canPreviousExperimentPage
                            onClicked: appController.compare.previousExperimentPage()
                        }
                        Text {
                            text: "Page " + appController.compare.experimentPageNumber
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                        LabButton {
                            text: "Next"
                            enabled: appController.compare.canNextExperimentPage
                            onClicked: appController.compare.nextExperimentPage()
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: experimentList.count + " shown"
                            color: Theme.dim
                            font.pixelSize: 10
                        }
                    }
                }
            }

            Panel {
                SplitView.fillWidth: true
                SplitView.minimumWidth: 400

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true

                        Text {
                            text: "CASE EVIDENCE"
                            color: Theme.dim
                            font.pixelSize: 10
                            font.weight: Font.Bold
                        }

                        Item { Layout.fillWidth: true }

                        CheckBox {
                            text: "Incorrect only"
                            checked: appController.compare.onlyIncorrect
                            enabled: !appController.compare.busy
                            activeFocusOnTab: true
                            onToggled: appController.compare.setOnlyIncorrect(checked)
                        }

                        Text {
                            text: appController.compare.caseTotal + " cases"
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: 28
                        color: Theme.surfaceAlt
                        radius: 5

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 8
                            anchors.rightMargin: 8
                            Text { Layout.preferredWidth: 150; text: "Example"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                            Text { Layout.preferredWidth: 62; text: "Result"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                            Text { Layout.preferredWidth: 72; text: "Latency"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                            Text { Layout.fillWidth: true; text: "Observed"; color: Theme.dim; font.pixelSize: 10; font.weight: Font.Bold }
                        }
                    }

                    ListView {
                        id: caseList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 120
                        clip: true
                        spacing: 2
                        model: appController.compare.cases

                        delegate: ItemDelegate {
                            required property var modelData
                            width: caseList.width
                            height: 42
                            activeFocusOnTab: true
                            Accessible.name: "Case " + modelData.exampleId
                            onClicked: appController.compare.selectCase(modelData.id)

                            contentItem: RowLayout {
                                spacing: 8
                                Text {
                                    Layout.preferredWidth: 150
                                    text: modelData.exampleId
                                    color: Theme.text2
                                    font.pixelSize: 11
                                    elide: Text.ElideMiddle
                                }
                                StatusPill {
                                    Layout.preferredWidth: 62
                                    text: modelData.correct ? "PASS" : "FAIL"
                                    tone: modelData.correct ? "good" : "bad"
                                }
                                Text {
                                    Layout.preferredWidth: 72
                                    text: Number(modelData.latencyMs).toFixed(2) + " ms"
                                    color: Theme.muted
                                    font.pixelSize: 10
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.observedPreview
                                    color: Theme.muted
                                    font.pixelSize: 10
                                    elide: Text.ElideRight
                                }
                            }

                            background: Rectangle {
                                radius: 5
                                color: appController.compare.selectedCase.id === modelData.id ?
                                       Theme.accentSurface : (parent.hovered ? Theme.hover : "transparent")
                                border.color: parent.activeFocus ? Theme.accent : "transparent"
                                border.width: parent.activeFocus ? 2 : 1
                            }
                        }

                        Text {
                            anchors.centerIn: parent
                            visible: caseList.count === 0
                            text: appController.compare.selectedExperiment.id ?
                                  "No evidence for this split/filter yet." : "Select an experiment."
                            color: Theme.dim
                            font.pixelSize: 12
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        LabButton {
                            text: "Previous"
                            enabled: appController.compare.canPreviousPage
                            onClicked: appController.compare.previousPage()
                        }
                        Text {
                            text: "Page " + appController.compare.pageNumber
                            color: Theme.muted
                            font.pixelSize: 11
                        }
                        LabButton {
                            text: "Next"
                            enabled: appController.compare.canNextPage
                            onClicked: appController.compare.nextPage()
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: appController.compare.selectedCase.failureId ?
                                  "Failure " + appController.compare.selectedCase.failureId.substring(0, 8) : ""
                            color: Theme.badText
                            font.pixelSize: 10
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: appController.compare.selectedCase.id ? 155 : 0
                        visible: appController.compare.selectedCase.id
                        radius: 7
                        color: Theme.surfaceAlt
                        border.color: Theme.border
                        clip: true

                        RowLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            spacing: 8

                            ColumnLayout {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Text { text: "EXPECTED"; color: Theme.dim; font.pixelSize: 9; font.weight: Font.Bold }
                                ScrollView {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    TextArea {
                                        text: appController.compare.selectedCase.expected || ""
                                        readOnly: true
                                        selectByMouse: true
                                        wrapMode: TextEdit.NoWrap
                                        color: Theme.text2
                                        font.family: "Consolas"
                                        font.pixelSize: 10
                                        background: null
                                    }
                                }
                            }

                            Rectangle { Layout.fillHeight: true; width: 1; color: Theme.border }

                            ColumnLayout {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Text { text: "OBSERVED"; color: Theme.dim; font.pixelSize: 9; font.weight: Font.Bold }
                                ScrollView {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    TextArea {
                                        text: appController.compare.selectedCase.observed || ""
                                        readOnly: true
                                        selectByMouse: true
                                        wrapMode: TextEdit.NoWrap
                                        color: Theme.text2
                                        font.family: "Consolas"
                                        font.pixelSize: 10
                                        background: null
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
