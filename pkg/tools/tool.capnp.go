// Code generated by capnpc-go. DO NOT EDIT.

package tools

import (
	capnp "capnproto.org/go/capnp/v3"
	text "capnproto.org/go/capnp/v3/encoding/text"
	schemas "capnproto.org/go/capnp/v3/schemas"
)

type Tool capnp.Struct

// Tool_TypeID is the unique identifier for the type Tool.
const Tool_TypeID = 0xfaf8531d9ac7b460

func NewTool(s *capnp.Segment) (Tool, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 1})
	return Tool(st), err
}

func NewRootTool(s *capnp.Segment) (Tool, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 1})
	return Tool(st), err
}

func ReadRootTool(msg *capnp.Message) (Tool, error) {
	root, err := msg.Root()
	return Tool(root.Struct()), err
}

func (s Tool) String() string {
	str, _ := text.Marshal(0xfaf8531d9ac7b460, capnp.Struct(s))
	return str
}

func (s Tool) EncodeAsPtr(seg *capnp.Segment) capnp.Ptr {
	return capnp.Struct(s).EncodeAsPtr(seg)
}

func (Tool) DecodeFromPtr(p capnp.Ptr) Tool {
	return Tool(capnp.Struct{}.DecodeFromPtr(p))
}

func (s Tool) ToPtr() capnp.Ptr {
	return capnp.Struct(s).ToPtr()
}
func (s Tool) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Tool) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Tool) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Tool) Function() (Function, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return Function(p.Struct()), err
}

func (s Tool) HasFunction() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Tool) SetFunction(v Function) error {
	return capnp.Struct(s).SetPtr(0, capnp.Struct(v).ToPtr())
}

// NewFunction sets the function field to a newly
// allocated Function struct, preferring placement in s's segment.
func (s Tool) NewFunction() (Function, error) {
	ss, err := NewFunction(capnp.Struct(s).Segment())
	if err != nil {
		return Function{}, err
	}
	err = capnp.Struct(s).SetPtr(0, capnp.Struct(ss).ToPtr())
	return ss, err
}

// Tool_List is a list of Tool.
type Tool_List = capnp.StructList[Tool]

// NewTool creates a new list of Tool.
func NewTool_List(s *capnp.Segment, sz int32) (Tool_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 0, PointerCount: 1}, sz)
	return capnp.StructList[Tool](l), err
}

// Tool_Future is a wrapper for a Tool promised by a client call.
type Tool_Future struct{ *capnp.Future }

func (f Tool_Future) Struct() (Tool, error) {
	p, err := f.Future.Ptr()
	return Tool(p.Struct()), err
}
func (p Tool_Future) Function() Function_Future {
	return Function_Future{Future: p.Future.Field(0, nil)}
}

type Function capnp.Struct

// Function_TypeID is the unique identifier for the type Function.
const Function_TypeID = 0xddabe0e1c16d3ea0

func NewFunction(s *capnp.Segment) (Function, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 3})
	return Function(st), err
}

func NewRootFunction(s *capnp.Segment) (Function, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 3})
	return Function(st), err
}

func ReadRootFunction(msg *capnp.Message) (Function, error) {
	root, err := msg.Root()
	return Function(root.Struct()), err
}

func (s Function) String() string {
	str, _ := text.Marshal(0xddabe0e1c16d3ea0, capnp.Struct(s))
	return str
}

func (s Function) EncodeAsPtr(seg *capnp.Segment) capnp.Ptr {
	return capnp.Struct(s).EncodeAsPtr(seg)
}

func (Function) DecodeFromPtr(p capnp.Ptr) Function {
	return Function(capnp.Struct{}.DecodeFromPtr(p))
}

func (s Function) ToPtr() capnp.Ptr {
	return capnp.Struct(s).ToPtr()
}
func (s Function) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Function) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Function) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Function) Name() (string, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.Text(), err
}

func (s Function) HasName() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Function) NameBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.TextBytes(), err
}

func (s Function) SetName(v string) error {
	return capnp.Struct(s).SetText(0, v)
}

func (s Function) Description() (string, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return p.Text(), err
}

func (s Function) HasDescription() bool {
	return capnp.Struct(s).HasPtr(1)
}

func (s Function) DescriptionBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return p.TextBytes(), err
}

func (s Function) SetDescription(v string) error {
	return capnp.Struct(s).SetText(1, v)
}

func (s Function) Parameters() (Parameters, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return Parameters(p.Struct()), err
}

func (s Function) HasParameters() bool {
	return capnp.Struct(s).HasPtr(2)
}

func (s Function) SetParameters(v Parameters) error {
	return capnp.Struct(s).SetPtr(2, capnp.Struct(v).ToPtr())
}

// NewParameters sets the parameters field to a newly
// allocated Parameters struct, preferring placement in s's segment.
func (s Function) NewParameters() (Parameters, error) {
	ss, err := NewParameters(capnp.Struct(s).Segment())
	if err != nil {
		return Parameters{}, err
	}
	err = capnp.Struct(s).SetPtr(2, capnp.Struct(ss).ToPtr())
	return ss, err
}

// Function_List is a list of Function.
type Function_List = capnp.StructList[Function]

// NewFunction creates a new list of Function.
func NewFunction_List(s *capnp.Segment, sz int32) (Function_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 0, PointerCount: 3}, sz)
	return capnp.StructList[Function](l), err
}

// Function_Future is a wrapper for a Function promised by a client call.
type Function_Future struct{ *capnp.Future }

func (f Function_Future) Struct() (Function, error) {
	p, err := f.Future.Ptr()
	return Function(p.Struct()), err
}
func (p Function_Future) Parameters() Parameters_Future {
	return Parameters_Future{Future: p.Future.Field(2, nil)}
}

type Parameters capnp.Struct

// Parameters_TypeID is the unique identifier for the type Parameters.
const Parameters_TypeID = 0xd5940dab0430c6b5

func NewParameters(s *capnp.Segment) (Parameters, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 3})
	return Parameters(st), err
}

func NewRootParameters(s *capnp.Segment) (Parameters, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 3})
	return Parameters(st), err
}

func ReadRootParameters(msg *capnp.Message) (Parameters, error) {
	root, err := msg.Root()
	return Parameters(root.Struct()), err
}

func (s Parameters) String() string {
	str, _ := text.Marshal(0xd5940dab0430c6b5, capnp.Struct(s))
	return str
}

func (s Parameters) EncodeAsPtr(seg *capnp.Segment) capnp.Ptr {
	return capnp.Struct(s).EncodeAsPtr(seg)
}

func (Parameters) DecodeFromPtr(p capnp.Ptr) Parameters {
	return Parameters(capnp.Struct{}.DecodeFromPtr(p))
}

func (s Parameters) ToPtr() capnp.Ptr {
	return capnp.Struct(s).ToPtr()
}
func (s Parameters) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Parameters) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Parameters) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Parameters) Type() (string, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.TextDefault("object"), err
}

func (s Parameters) HasType() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Parameters) TypeBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.TextBytesDefault("object"), err
}

func (s Parameters) SetType(v string) error {
	return capnp.Struct(s).SetNewText(0, v)
}

func (s Parameters) Properties() (Property_List, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return Property_List(p.List()), err
}

func (s Parameters) HasProperties() bool {
	return capnp.Struct(s).HasPtr(1)
}

func (s Parameters) SetProperties(v Property_List) error {
	return capnp.Struct(s).SetPtr(1, v.ToPtr())
}

// NewProperties sets the properties field to a newly
// allocated Property_List, preferring placement in s's segment.
func (s Parameters) NewProperties(n int32) (Property_List, error) {
	l, err := NewProperty_List(capnp.Struct(s).Segment(), n)
	if err != nil {
		return Property_List{}, err
	}
	err = capnp.Struct(s).SetPtr(1, l.ToPtr())
	return l, err
}
func (s Parameters) Required() (capnp.TextList, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return capnp.TextList(p.List()), err
}

func (s Parameters) HasRequired() bool {
	return capnp.Struct(s).HasPtr(2)
}

func (s Parameters) SetRequired(v capnp.TextList) error {
	return capnp.Struct(s).SetPtr(2, v.ToPtr())
}

// NewRequired sets the required field to a newly
// allocated capnp.TextList, preferring placement in s's segment.
func (s Parameters) NewRequired(n int32) (capnp.TextList, error) {
	l, err := capnp.NewTextList(capnp.Struct(s).Segment(), n)
	if err != nil {
		return capnp.TextList{}, err
	}
	err = capnp.Struct(s).SetPtr(2, l.ToPtr())
	return l, err
}

// Parameters_List is a list of Parameters.
type Parameters_List = capnp.StructList[Parameters]

// NewParameters creates a new list of Parameters.
func NewParameters_List(s *capnp.Segment, sz int32) (Parameters_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 0, PointerCount: 3}, sz)
	return capnp.StructList[Parameters](l), err
}

// Parameters_Future is a wrapper for a Parameters promised by a client call.
type Parameters_Future struct{ *capnp.Future }

func (f Parameters_Future) Struct() (Parameters, error) {
	p, err := f.Future.Ptr()
	return Parameters(p.Struct()), err
}

type Property capnp.Struct

// Property_TypeID is the unique identifier for the type Property.
const Property_TypeID = 0xe667140e5823722d

func NewProperty(s *capnp.Segment) (Property, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 4})
	return Property(st), err
}

func NewRootProperty(s *capnp.Segment) (Property, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 4})
	return Property(st), err
}

func ReadRootProperty(msg *capnp.Message) (Property, error) {
	root, err := msg.Root()
	return Property(root.Struct()), err
}

func (s Property) String() string {
	str, _ := text.Marshal(0xe667140e5823722d, capnp.Struct(s))
	return str
}

func (s Property) EncodeAsPtr(seg *capnp.Segment) capnp.Ptr {
	return capnp.Struct(s).EncodeAsPtr(seg)
}

func (Property) DecodeFromPtr(p capnp.Ptr) Property {
	return Property(capnp.Struct{}.DecodeFromPtr(p))
}

func (s Property) ToPtr() capnp.Ptr {
	return capnp.Struct(s).ToPtr()
}
func (s Property) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Property) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Property) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Property) Name() (string, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.Text(), err
}

func (s Property) HasName() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Property) NameBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.TextBytes(), err
}

func (s Property) SetName(v string) error {
	return capnp.Struct(s).SetText(0, v)
}

func (s Property) Type() (string, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return p.Text(), err
}

func (s Property) HasType() bool {
	return capnp.Struct(s).HasPtr(1)
}

func (s Property) TypeBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return p.TextBytes(), err
}

func (s Property) SetType(v string) error {
	return capnp.Struct(s).SetText(1, v)
}

func (s Property) Description() (string, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return p.Text(), err
}

func (s Property) HasDescription() bool {
	return capnp.Struct(s).HasPtr(2)
}

func (s Property) DescriptionBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return p.TextBytes(), err
}

func (s Property) SetDescription(v string) error {
	return capnp.Struct(s).SetText(2, v)
}

func (s Property) Enum() (capnp.TextList, error) {
	p, err := capnp.Struct(s).Ptr(3)
	return capnp.TextList(p.List()), err
}

func (s Property) HasEnum() bool {
	return capnp.Struct(s).HasPtr(3)
}

func (s Property) SetEnum(v capnp.TextList) error {
	return capnp.Struct(s).SetPtr(3, v.ToPtr())
}

// NewEnum sets the enum field to a newly
// allocated capnp.TextList, preferring placement in s's segment.
func (s Property) NewEnum(n int32) (capnp.TextList, error) {
	l, err := capnp.NewTextList(capnp.Struct(s).Segment(), n)
	if err != nil {
		return capnp.TextList{}, err
	}
	err = capnp.Struct(s).SetPtr(3, l.ToPtr())
	return l, err
}

// Property_List is a list of Property.
type Property_List = capnp.StructList[Property]

// NewProperty creates a new list of Property.
func NewProperty_List(s *capnp.Segment, sz int32) (Property_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 0, PointerCount: 4}, sz)
	return capnp.StructList[Property](l), err
}

// Property_Future is a wrapper for a Property promised by a client call.
type Property_Future struct{ *capnp.Future }

func (f Property_Future) Struct() (Property, error) {
	p, err := f.Future.Ptr()
	return Property(p.Struct()), err
}

const schema_d4c9c9f76e88a0d1 = "x\xda\x8c\xd2\xbfo\xd3^\x14\x05\xf0s\xdeu\xbfY" +
	"\x92&\x96S\xe9\xbbT\x05\xb1!\xb5\x14\xb1u\xa0]" +
	"@\x14\x15)O\x14\x09\xa1\x0eM\x93GkHl\xf7" +
	"\xd9\x19\xba1\xf6\x0f`b\xea^!\x16:000" +
	"@\x95\x11\x04#\x12,\xb0w\x01\xb1\x18=\xd3\xfc " +
	"j%&[\xd7G~\x9f{\xef[\\\xe0\x8aw\xb5" +
	"r\xac\xa0\xf4\x85\xa9\xff\xf2\xa3w\x8b\xdea\xe5\xe9'" +
	"\xf83\xcc\xdf\x1f\xecG?\xfa\xfd\x8f\x98\x92\x12pm" +
	"\x95\x8a\xc1=\x96\x80@\xf3\x05\x98\x1f\\\xef\xbe\xf9\xfa" +
	"\xe5\xf0\xf3\x19\xe1`F\x9d\x04\x17\x95{\x9bU.;" +
	"o/\xdd\x9f\xaeo\x7f\x9b\xc8z.\xf1Z\x9d\x04\xfd" +
	"\"\xfbV}\x07\xf3\xcd\x97\xc7\xcff\xef\xfe\xfc5\x91" +
	"-N~.\x1f\x82W\xc5\x09G\xb2\x8c\xf9<y\xbc" +
	"}%\x8b\xe3\x8e\xa4\xc5c\xa1\xd5L\xa2d\xa9\xd1\xb4" +
	"\xcd\xae\xc9\x8cM\xd1 uY<\xc0#\xe9\xdf\xb8\x0c" +
	"\xe8\x15\xa1^S\xf4\xc9:\x09\xf8w\x1e\x00zM\xa8" +
	"w\x14}\xa5\xeaT\x80on\x03\xba-\xd4O\x14\xab" +
	"\xd9^bX\x86b\x19\xf0\xb9\xb4\x1co=2\xad," +
	"Ol\x9c\x18\x9b\x85\x10\x93r\x1al\x08Y\x1b\xb5\x0a" +
	"\xbabn\xcdn/\xb4\xa6\x0d`\x10r\xbfr\x9f\x06" +
	"z\xf5\x97\xfef/\x9akea\x1c\x8d\xd9\x81\xb3\xed" +
	"\xab[\x80\xbe%\xd4\xebcv\xed\x1aj\x08\xf5\x86b" +
	"5jv\x07v\xe6m\x93\xb6l\x98d(\x85q4" +
	"\xac&\xa7\xe3\x82\xd8\x94\xb5\xd1%\x00Y;W\xd9\xb0" +
	"\xf1\x9c\xeb~\xcf)kCe\xd3)7N\x879P" +
	"\x1aW\xdc\x14\xea\xce\x982t\xf4\x1d\xa1\xce\x14}\x91" +
	":\x05\xf0w]\xb2#\xd4\xfb\x13\xf4\xf1\x1d\x9c\xd3G" +
	"\xd5D\xbd\xee\xbf\x8dx=\x8e\xd9qpo\x08\xaf\xb8" +
	"\x85\x97\x85\xfa\x7f\xc5\xfca/*6\xe0vV\x1b]" +
	"\xf4?\x03\xf9\x1d\x00\x00\xff\xff\x1d\xda\xc5$"

func RegisterSchema(reg *schemas.Registry) {
	reg.Register(&schemas.Schema{
		String: schema_d4c9c9f76e88a0d1,
		Nodes: []uint64{
			0xd5940dab0430c6b5,
			0xddabe0e1c16d3ea0,
			0xe667140e5823722d,
			0xfaf8531d9ac7b460,
		},
		Compressed: true,
	})
}
