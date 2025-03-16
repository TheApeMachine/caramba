// Code generated by capnpc-go. DO NOT EDIT.

package message

import (
	capnp "capnproto.org/go/capnp/v3"
	text "capnproto.org/go/capnp/v3/encoding/text"
	schemas "capnproto.org/go/capnp/v3/schemas"
)

type Artifact capnp.Struct

// Artifact_TypeID is the unique identifier for the type Artifact.
const Artifact_TypeID = 0x9cc399b32bed09b5

func NewArtifact(s *capnp.Segment) (Artifact, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 4})
	return Artifact(st), err
}

func NewRootArtifact(s *capnp.Segment) (Artifact, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 4})
	return Artifact(st), err
}

func ReadRootArtifact(msg *capnp.Message) (Artifact, error) {
	root, err := msg.Root()
	return Artifact(root.Struct()), err
}

func (s Artifact) String() string {
	str, _ := text.Marshal(0x9cc399b32bed09b5, capnp.Struct(s))
	return str
}

func (s Artifact) EncodeAsPtr(seg *capnp.Segment) capnp.Ptr {
	return capnp.Struct(s).EncodeAsPtr(seg)
}

func (Artifact) DecodeFromPtr(p capnp.Ptr) Artifact {
	return Artifact(capnp.Struct{}.DecodeFromPtr(p))
}

func (s Artifact) ToPtr() capnp.Ptr {
	return capnp.Struct(s).ToPtr()
}
func (s Artifact) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Artifact) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Artifact) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Artifact) Id() (string, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.Text(), err
}

func (s Artifact) HasId() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Artifact) IdBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.TextBytes(), err
}

func (s Artifact) SetId(v string) error {
	return capnp.Struct(s).SetText(0, v)
}

func (s Artifact) Role() (string, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return p.Text(), err
}

func (s Artifact) HasRole() bool {
	return capnp.Struct(s).HasPtr(1)
}

func (s Artifact) RoleBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return p.TextBytes(), err
}

func (s Artifact) SetRole(v string) error {
	return capnp.Struct(s).SetText(1, v)
}

func (s Artifact) Name() (string, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return p.Text(), err
}

func (s Artifact) HasName() bool {
	return capnp.Struct(s).HasPtr(2)
}

func (s Artifact) NameBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return p.TextBytes(), err
}

func (s Artifact) SetName(v string) error {
	return capnp.Struct(s).SetText(2, v)
}

func (s Artifact) Content() (string, error) {
	p, err := capnp.Struct(s).Ptr(3)
	return p.Text(), err
}

func (s Artifact) HasContent() bool {
	return capnp.Struct(s).HasPtr(3)
}

func (s Artifact) ContentBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(3)
	return p.TextBytes(), err
}

func (s Artifact) SetContent(v string) error {
	return capnp.Struct(s).SetText(3, v)
}

// Artifact_List is a list of Artifact.
type Artifact_List = capnp.StructList[Artifact]

// NewArtifact creates a new list of Artifact.
func NewArtifact_List(s *capnp.Segment, sz int32) (Artifact_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 0, PointerCount: 4}, sz)
	return capnp.StructList[Artifact](l), err
}

// Artifact_Future is a wrapper for a Artifact promised by a client call.
type Artifact_Future struct{ *capnp.Future }

func (f Artifact_Future) Struct() (Artifact, error) {
	p, err := f.Future.Ptr()
	return Artifact(p.Struct()), err
}

const schema_e363a5839bf866c5 = "x\xda<\xc8!N\x03A\x14\x06\xe0\xff\x9f\xb7eC" +
	"RB_\xb2\x0a\x1c\x12\x02M-\x0a8\x01\xcf#\x98" +
	"L\xa7\xed\x86\xeev\xd3\x8e\x87\x84S\x10\xb0\x9c\x80\xa0" +
	"\x11\xa4\x07@p\x02\x04\x0a\x8f[\xb2\x82\xca\xef\x1b\xdc" +
	"\x9de\xa3\x9d7\xc2Y\xd1\xdbj_\xb7\x7f\x8e^\x1e" +
	"\xde\x9f\xa0\xfbl\xd7\x93\xdf\xc7\xfb\xe7\xf0\x85^\x96\x03" +
	"\xa3\xf5\x01\xf53\x07\xf4\xe3\x1b\xc7ms3\x1dVq" +
	"\xb5\x12?\x8dC\xbfL\xe5\xc4\x87t\x12|S7\xa7" +
	"\xe7\x1ds\x1f\xd2%i\x03\xc9\x80\x8c\x80\xfa=\xc0\xae" +
	"\x846sT\xb2`\x97\xf1\x10\xb0k\xa1\xcd\x1d\xd5\xb9" +
	"\x82\x0e\xd0\xb2\xcb\xb1\xd0\x1aG\x15)(\x80V\x17\x80" +
	"\xcd\x84\x96\x1c\xa5\x1c\xb3\x0f\xc7>\xb8\xbb\\\xcc\xe3\x06" +
	"\xb5\xaf6\xb8\x0d\x8b:\xc5:\xfd\xfb/\x00\x00\xff\xff" +
	"j\x802\x93"

func RegisterSchema(reg *schemas.Registry) {
	reg.Register(&schemas.Schema{
		String: schema_e363a5839bf866c5,
		Nodes: []uint64{
			0x9cc399b32bed09b5,
		},
		Compressed: true,
	})
}
