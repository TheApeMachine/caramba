import React from "react";
import styles from "./Control.css";
import Checkbox from "../Checkbox/Checkbox";
import TextInput from "../TextInput/TextInput";
import Select from "../Select/Select";
import { NodeDispatchContext, ContextContext } from "../../context";
import type {
    ControlData,
    ControlRenderCallback,
    ControlTypes,
    InputData,
    SelectOption,
    ValueSetter
} from "../../types";
import { NodesActionType } from "../../nodesReducer";

interface CommonProps {
    name: string;
    nodeId: string;
    portName: string;
    label: string;
    inputLabel: string;
    allData: ControlData;
    inputData: InputData;
    triggerRecalculation: () => void;
    updateNodeConnections: () => void;
    setValue?: ValueSetter;
    isMonoControl?: boolean;
}

interface TextInputProps extends CommonProps {
    type: "text";
    data: string;
    defaultValue?: string;
    placeholder?: string;
}

interface NumberInputProps extends CommonProps {
    type: "number";
    data: number;
    defaultValue?: number;
    step?: number;
    placeholder?: string;
}

interface CheckboxProps extends CommonProps {
    type: "checkbox";
    data: boolean;
    defaultValue?: boolean;
}

interface SelectProps extends CommonProps {
    type: "select";
    data: string;
    options: SelectOption[];
    defaultValue?: string;
    placeholder?: string;
    getOptions?: (
        inputData: InputData,
        executionContext: unknown
    ) => SelectOption[];
}

interface MultiSelectProps extends CommonProps {
    type: "multiselect";
    data: string[];
    options: SelectOption[];
    defaultValue?: string[];
    placeholder?: string;
    getOptions?: (
        inputData: InputData,
        executionContext: unknown
    ) => SelectOption[];
}

interface CustomProps extends CommonProps {
    type: "custom";
    data: unknown;
    defaultValue?: unknown;
    render?: ControlRenderCallback;
}

type ControlProps =
    | TextInputProps
    | NumberInputProps
    | CheckboxProps
    | SelectProps
    | MultiSelectProps
    | CustomProps;

const Control = (props: ControlProps) => {
    const {
        type,
        name,
        nodeId,
        portName,
        label,
        inputLabel,
        data,
        allData,
        inputData,
        triggerRecalculation,
        updateNodeConnections,
        setValue,
        defaultValue,
        isMonoControl
    } = props;
    const nodesDispatch = React.useContext(NodeDispatchContext);
    const executionContext = React.useContext(ContextContext);

    const calculatedLabel = isMonoControl ? inputLabel : label;

    const onChange = (data: unknown) => {
        if (nodesDispatch) {
            nodesDispatch({
                type: NodesActionType.SET_PORT_DATA,
                data,
                nodeId,
                portName,
                controlName: name,
                setValue
            });
        }
        triggerRecalculation();
    };

    const getControlByType = (type: ControlTypes) => {
        const commonProps = {
            triggerRecalculation,
            updateNodeConnections,
            onChange
        } as const;
        switch (type) {
            case "select": {
                const { options, getOptions, placeholder } =
                    props as SelectProps;

                return (
                    <Select
                        onChange={
                            commonProps.onChange as (
                                d: string | string[]
                            ) => void
                        }
                        data={props.data as string}
                        options={
                            getOptions
                                ? getOptions(inputData, executionContext)
                                : options
                        }
                        placeholder={placeholder}
                    />
                );
            }
            case "text": {
                const { placeholder } = props as TextInputProps;

                return (
                    <TextInput
                        updateNodeConnections={updateNodeConnections}
                        onChange={
                            commonProps.onChange as (d: string | number) => void
                        }
                        data={props.data as string}
                        placeholder={placeholder}
                    />
                );
            }
            case "number": {
                const { step, placeholder } = props as NumberInputProps;

                return (
                    <TextInput
                        updateNodeConnections={updateNodeConnections}
                        onChange={
                            commonProps.onChange as (d: string | number) => void
                        }
                        step={step}
                        type="number"
                        data={props.data as number}
                        placeholder={placeholder}
                    />
                );
            }
            case "checkbox":
                return (
                    <Checkbox
                        onChange={commonProps.onChange as (d: boolean) => void}
                        data={props.data as boolean}
                        label={calculatedLabel}
                    />
                );
            case "multiselect": {
                const { options, getOptions, placeholder } =
                    props as MultiSelectProps;

                return (
                    <Select
                        allowMultiple
                        onChange={
                            commonProps.onChange as (
                                d: string | string[]
                            ) => void
                        }
                        data={props.data as string[]}
                        options={
                            getOptions
                                ? getOptions(inputData, executionContext)
                                : options
                        }
                        placeholder={placeholder}
                    />
                );
            }
            case "custom": {
                const { render } = props as CustomProps;

                return (
                    render?.(
                        data,
                        onChange,
                        executionContext,
                        triggerRecalculation,
                        {
                            label,
                            name,
                            portName,
                            inputLabel,
                            defaultValue
                        },
                        allData
                    ) ?? null
                );
            }
            default:
                return <div>Control</div>;
        }
    };

    return (
        <div className={styles.wrapper} data-flume-component="control">
            {calculatedLabel && type !== "checkbox" && type !== "custom" && (
                <span
                    data-flume-component="control-label"
                    className={styles.controlLabel}
                >
                    {calculatedLabel}
                </span>
            )}
            {getControlByType(type)}
        </div>
    );
};

export default Control;
