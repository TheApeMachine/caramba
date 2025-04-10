// Code generated by capnpc-go. DO NOT EDIT.

package datura

import (
	capnp "capnproto.org/go/capnp/v3"
	text "capnproto.org/go/capnp/v3/encoding/text"
	schemas "capnproto.org/go/capnp/v3/schemas"
	math "math"
	strconv "strconv"
)

type Artifact capnp.Struct

// Artifact_TypeID is the unique identifier for the type Artifact.
const Artifact_TypeID = 0xb1092b0e00ae75e5

func NewArtifact(s *capnp.Segment) (Artifact, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 24, PointerCount: 14})
	return Artifact(st), err
}

func NewRootArtifact(s *capnp.Segment) (Artifact, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 24, PointerCount: 14})
	return Artifact(st), err
}

func ReadRootArtifact(msg *capnp.Message) (Artifact, error) {
	root, err := msg.Root()
	return Artifact(root.Struct()), err
}

func (s Artifact) String() string {
	str, _ := text.Marshal(0xb1092b0e00ae75e5, capnp.Struct(s))
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
func (s Artifact) Uuid() (string, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.Text(), err
}

func (s Artifact) HasUuid() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Artifact) UuidBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.TextBytes(), err
}

func (s Artifact) SetUuid(v string) error {
	return capnp.Struct(s).SetText(0, v)
}

func (s Artifact) State() uint64 {
	return capnp.Struct(s).Uint64(0)
}

func (s Artifact) SetState(v uint64) {
	capnp.Struct(s).SetUint64(0, v)
}

func (s Artifact) Checksum() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return []byte(p.Data()), err
}

func (s Artifact) HasChecksum() bool {
	return capnp.Struct(s).HasPtr(1)
}

func (s Artifact) SetChecksum(v []byte) error {
	return capnp.Struct(s).SetData(1, v)
}

func (s Artifact) Timestamp() int64 {
	return int64(capnp.Struct(s).Uint64(8))
}

func (s Artifact) SetTimestamp(v int64) {
	capnp.Struct(s).SetUint64(8, uint64(v))
}

func (s Artifact) Mediatype() (string, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return p.Text(), err
}

func (s Artifact) HasMediatype() bool {
	return capnp.Struct(s).HasPtr(2)
}

func (s Artifact) MediatypeBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(2)
	return p.TextBytes(), err
}

func (s Artifact) SetMediatype(v string) error {
	return capnp.Struct(s).SetText(2, v)
}

func (s Artifact) Origin() (string, error) {
	p, err := capnp.Struct(s).Ptr(3)
	return p.Text(), err
}

func (s Artifact) HasOrigin() bool {
	return capnp.Struct(s).HasPtr(3)
}

func (s Artifact) OriginBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(3)
	return p.TextBytes(), err
}

func (s Artifact) SetOrigin(v string) error {
	return capnp.Struct(s).SetText(3, v)
}

func (s Artifact) Issuer() (string, error) {
	p, err := capnp.Struct(s).Ptr(4)
	return p.Text(), err
}

func (s Artifact) HasIssuer() bool {
	return capnp.Struct(s).HasPtr(4)
}

func (s Artifact) IssuerBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(4)
	return p.TextBytes(), err
}

func (s Artifact) SetIssuer(v string) error {
	return capnp.Struct(s).SetText(4, v)
}

func (s Artifact) Role() uint32 {
	return capnp.Struct(s).Uint32(16)
}

func (s Artifact) SetRole(v uint32) {
	capnp.Struct(s).SetUint32(16, v)
}

func (s Artifact) Scope() uint32 {
	return capnp.Struct(s).Uint32(20)
}

func (s Artifact) SetScope(v uint32) {
	capnp.Struct(s).SetUint32(20, v)
}

func (s Artifact) PseudonymHash() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(5)
	return []byte(p.Data()), err
}

func (s Artifact) HasPseudonymHash() bool {
	return capnp.Struct(s).HasPtr(5)
}

func (s Artifact) SetPseudonymHash(v []byte) error {
	return capnp.Struct(s).SetData(5, v)
}

func (s Artifact) MerkleRoot() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(6)
	return []byte(p.Data()), err
}

func (s Artifact) HasMerkleRoot() bool {
	return capnp.Struct(s).HasPtr(6)
}

func (s Artifact) SetMerkleRoot(v []byte) error {
	return capnp.Struct(s).SetData(6, v)
}

func (s Artifact) Metadata() (Artifact_Metadata_List, error) {
	p, err := capnp.Struct(s).Ptr(7)
	return Artifact_Metadata_List(p.List()), err
}

func (s Artifact) HasMetadata() bool {
	return capnp.Struct(s).HasPtr(7)
}

func (s Artifact) SetMetadata(v Artifact_Metadata_List) error {
	return capnp.Struct(s).SetPtr(7, v.ToPtr())
}

// NewMetadata sets the metadata field to a newly
// allocated Artifact_Metadata_List, preferring placement in s's segment.
func (s Artifact) NewMetadata(n int32) (Artifact_Metadata_List, error) {
	l, err := NewArtifact_Metadata_List(capnp.Struct(s).Segment(), n)
	if err != nil {
		return Artifact_Metadata_List{}, err
	}
	err = capnp.Struct(s).SetPtr(7, l.ToPtr())
	return l, err
}
func (s Artifact) Payload() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(8)
	return []byte(p.Data()), err
}

func (s Artifact) HasPayload() bool {
	return capnp.Struct(s).HasPtr(8)
}

func (s Artifact) SetPayload(v []byte) error {
	return capnp.Struct(s).SetData(8, v)
}

func (s Artifact) EncryptedPayload() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(9)
	return []byte(p.Data()), err
}

func (s Artifact) HasEncryptedPayload() bool {
	return capnp.Struct(s).HasPtr(9)
}

func (s Artifact) SetEncryptedPayload(v []byte) error {
	return capnp.Struct(s).SetData(9, v)
}

func (s Artifact) EncryptedKey() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(10)
	return []byte(p.Data()), err
}

func (s Artifact) HasEncryptedKey() bool {
	return capnp.Struct(s).HasPtr(10)
}

func (s Artifact) SetEncryptedKey(v []byte) error {
	return capnp.Struct(s).SetData(10, v)
}

func (s Artifact) EphemeralPublicKey() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(11)
	return []byte(p.Data()), err
}

func (s Artifact) HasEphemeralPublicKey() bool {
	return capnp.Struct(s).HasPtr(11)
}

func (s Artifact) SetEphemeralPublicKey(v []byte) error {
	return capnp.Struct(s).SetData(11, v)
}

func (s Artifact) Approvals() (Artifact_Approval_List, error) {
	p, err := capnp.Struct(s).Ptr(12)
	return Artifact_Approval_List(p.List()), err
}

func (s Artifact) HasApprovals() bool {
	return capnp.Struct(s).HasPtr(12)
}

func (s Artifact) SetApprovals(v Artifact_Approval_List) error {
	return capnp.Struct(s).SetPtr(12, v.ToPtr())
}

// NewApprovals sets the approvals field to a newly
// allocated Artifact_Approval_List, preferring placement in s's segment.
func (s Artifact) NewApprovals(n int32) (Artifact_Approval_List, error) {
	l, err := NewArtifact_Approval_List(capnp.Struct(s).Segment(), n)
	if err != nil {
		return Artifact_Approval_List{}, err
	}
	err = capnp.Struct(s).SetPtr(12, l.ToPtr())
	return l, err
}
func (s Artifact) Signature() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(13)
	return []byte(p.Data()), err
}

func (s Artifact) HasSignature() bool {
	return capnp.Struct(s).HasPtr(13)
}

func (s Artifact) SetSignature(v []byte) error {
	return capnp.Struct(s).SetData(13, v)
}

// Artifact_List is a list of Artifact.
type Artifact_List = capnp.StructList[Artifact]

// NewArtifact creates a new list of Artifact.
func NewArtifact_List(s *capnp.Segment, sz int32) (Artifact_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 24, PointerCount: 14}, sz)
	return capnp.StructList[Artifact](l), err
}

// Artifact_Future is a wrapper for a Artifact promised by a client call.
type Artifact_Future struct{ *capnp.Future }

func (f Artifact_Future) Struct() (Artifact, error) {
	p, err := f.Future.Ptr()
	return Artifact(p.Struct()), err
}

type Artifact_Metadata capnp.Struct
type Artifact_Metadata_value Artifact_Metadata
type Artifact_Metadata_value_Which uint16

const (
	Artifact_Metadata_value_Which_textValue   Artifact_Metadata_value_Which = 0
	Artifact_Metadata_value_Which_intValue    Artifact_Metadata_value_Which = 1
	Artifact_Metadata_value_Which_floatValue  Artifact_Metadata_value_Which = 2
	Artifact_Metadata_value_Which_boolValue   Artifact_Metadata_value_Which = 3
	Artifact_Metadata_value_Which_binaryValue Artifact_Metadata_value_Which = 4
)

func (w Artifact_Metadata_value_Which) String() string {
	const s = "textValueintValuefloatValueboolValuebinaryValue"
	switch w {
	case Artifact_Metadata_value_Which_textValue:
		return s[0:9]
	case Artifact_Metadata_value_Which_intValue:
		return s[9:17]
	case Artifact_Metadata_value_Which_floatValue:
		return s[17:27]
	case Artifact_Metadata_value_Which_boolValue:
		return s[27:36]
	case Artifact_Metadata_value_Which_binaryValue:
		return s[36:47]

	}
	return "Artifact_Metadata_value_Which(" + strconv.FormatUint(uint64(w), 10) + ")"
}

// Artifact_Metadata_TypeID is the unique identifier for the type Artifact_Metadata.
const Artifact_Metadata_TypeID = 0xc121b65fcd3a05ee

func NewArtifact_Metadata(s *capnp.Segment) (Artifact_Metadata, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 16, PointerCount: 2})
	return Artifact_Metadata(st), err
}

func NewRootArtifact_Metadata(s *capnp.Segment) (Artifact_Metadata, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 16, PointerCount: 2})
	return Artifact_Metadata(st), err
}

func ReadRootArtifact_Metadata(msg *capnp.Message) (Artifact_Metadata, error) {
	root, err := msg.Root()
	return Artifact_Metadata(root.Struct()), err
}

func (s Artifact_Metadata) String() string {
	str, _ := text.Marshal(0xc121b65fcd3a05ee, capnp.Struct(s))
	return str
}

func (s Artifact_Metadata) EncodeAsPtr(seg *capnp.Segment) capnp.Ptr {
	return capnp.Struct(s).EncodeAsPtr(seg)
}

func (Artifact_Metadata) DecodeFromPtr(p capnp.Ptr) Artifact_Metadata {
	return Artifact_Metadata(capnp.Struct{}.DecodeFromPtr(p))
}

func (s Artifact_Metadata) ToPtr() capnp.Ptr {
	return capnp.Struct(s).ToPtr()
}
func (s Artifact_Metadata) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Artifact_Metadata) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Artifact_Metadata) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Artifact_Metadata) Key() (string, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.Text(), err
}

func (s Artifact_Metadata) HasKey() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Artifact_Metadata) KeyBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return p.TextBytes(), err
}

func (s Artifact_Metadata) SetKey(v string) error {
	return capnp.Struct(s).SetText(0, v)
}

func (s Artifact_Metadata) Value() Artifact_Metadata_value { return Artifact_Metadata_value(s) }

func (s Artifact_Metadata_value) Which() Artifact_Metadata_value_Which {
	return Artifact_Metadata_value_Which(capnp.Struct(s).Uint16(0))
}
func (s Artifact_Metadata_value) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Artifact_Metadata_value) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Artifact_Metadata_value) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Artifact_Metadata_value) TextValue() (string, error) {
	if capnp.Struct(s).Uint16(0) != 0 {
		panic("Which() != textValue")
	}
	p, err := capnp.Struct(s).Ptr(1)
	return p.Text(), err
}

func (s Artifact_Metadata_value) HasTextValue() bool {
	if capnp.Struct(s).Uint16(0) != 0 {
		return false
	}
	return capnp.Struct(s).HasPtr(1)
}

func (s Artifact_Metadata_value) TextValueBytes() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return p.TextBytes(), err
}

func (s Artifact_Metadata_value) SetTextValue(v string) error {
	capnp.Struct(s).SetUint16(0, 0)
	return capnp.Struct(s).SetText(1, v)
}

func (s Artifact_Metadata_value) IntValue() int64 {
	if capnp.Struct(s).Uint16(0) != 1 {
		panic("Which() != intValue")
	}
	return int64(capnp.Struct(s).Uint64(8))
}

func (s Artifact_Metadata_value) SetIntValue(v int64) {
	capnp.Struct(s).SetUint16(0, 1)
	capnp.Struct(s).SetUint64(8, uint64(v))
}

func (s Artifact_Metadata_value) FloatValue() float64 {
	if capnp.Struct(s).Uint16(0) != 2 {
		panic("Which() != floatValue")
	}
	return math.Float64frombits(capnp.Struct(s).Uint64(8))
}

func (s Artifact_Metadata_value) SetFloatValue(v float64) {
	capnp.Struct(s).SetUint16(0, 2)
	capnp.Struct(s).SetUint64(8, math.Float64bits(v))
}

func (s Artifact_Metadata_value) BoolValue() bool {
	if capnp.Struct(s).Uint16(0) != 3 {
		panic("Which() != boolValue")
	}
	return capnp.Struct(s).Bit(64)
}

func (s Artifact_Metadata_value) SetBoolValue(v bool) {
	capnp.Struct(s).SetUint16(0, 3)
	capnp.Struct(s).SetBit(64, v)
}

func (s Artifact_Metadata_value) BinaryValue() ([]byte, error) {
	if capnp.Struct(s).Uint16(0) != 4 {
		panic("Which() != binaryValue")
	}
	p, err := capnp.Struct(s).Ptr(1)
	return []byte(p.Data()), err
}

func (s Artifact_Metadata_value) HasBinaryValue() bool {
	if capnp.Struct(s).Uint16(0) != 4 {
		return false
	}
	return capnp.Struct(s).HasPtr(1)
}

func (s Artifact_Metadata_value) SetBinaryValue(v []byte) error {
	capnp.Struct(s).SetUint16(0, 4)
	return capnp.Struct(s).SetData(1, v)
}

// Artifact_Metadata_List is a list of Artifact_Metadata.
type Artifact_Metadata_List = capnp.StructList[Artifact_Metadata]

// NewArtifact_Metadata creates a new list of Artifact_Metadata.
func NewArtifact_Metadata_List(s *capnp.Segment, sz int32) (Artifact_Metadata_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 16, PointerCount: 2}, sz)
	return capnp.StructList[Artifact_Metadata](l), err
}

// Artifact_Metadata_Future is a wrapper for a Artifact_Metadata promised by a client call.
type Artifact_Metadata_Future struct{ *capnp.Future }

func (f Artifact_Metadata_Future) Struct() (Artifact_Metadata, error) {
	p, err := f.Future.Ptr()
	return Artifact_Metadata(p.Struct()), err
}
func (p Artifact_Metadata_Future) Value() Artifact_Metadata_value_Future {
	return Artifact_Metadata_value_Future{p.Future}
}

// Artifact_Metadata_value_Future is a wrapper for a Artifact_Metadata_value promised by a client call.
type Artifact_Metadata_value_Future struct{ *capnp.Future }

func (f Artifact_Metadata_value_Future) Struct() (Artifact_Metadata_value, error) {
	p, err := f.Future.Ptr()
	return Artifact_Metadata_value(p.Struct()), err
}

type Artifact_Approval capnp.Struct

// Artifact_Approval_TypeID is the unique identifier for the type Artifact_Approval.
const Artifact_Approval_TypeID = 0x8cc20228b0f1020d

func NewArtifact_Approval(s *capnp.Segment) (Artifact_Approval, error) {
	st, err := capnp.NewStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 2})
	return Artifact_Approval(st), err
}

func NewRootArtifact_Approval(s *capnp.Segment) (Artifact_Approval, error) {
	st, err := capnp.NewRootStruct(s, capnp.ObjectSize{DataSize: 0, PointerCount: 2})
	return Artifact_Approval(st), err
}

func ReadRootArtifact_Approval(msg *capnp.Message) (Artifact_Approval, error) {
	root, err := msg.Root()
	return Artifact_Approval(root.Struct()), err
}

func (s Artifact_Approval) String() string {
	str, _ := text.Marshal(0x8cc20228b0f1020d, capnp.Struct(s))
	return str
}

func (s Artifact_Approval) EncodeAsPtr(seg *capnp.Segment) capnp.Ptr {
	return capnp.Struct(s).EncodeAsPtr(seg)
}

func (Artifact_Approval) DecodeFromPtr(p capnp.Ptr) Artifact_Approval {
	return Artifact_Approval(capnp.Struct{}.DecodeFromPtr(p))
}

func (s Artifact_Approval) ToPtr() capnp.Ptr {
	return capnp.Struct(s).ToPtr()
}
func (s Artifact_Approval) IsValid() bool {
	return capnp.Struct(s).IsValid()
}

func (s Artifact_Approval) Message() *capnp.Message {
	return capnp.Struct(s).Message()
}

func (s Artifact_Approval) Segment() *capnp.Segment {
	return capnp.Struct(s).Segment()
}
func (s Artifact_Approval) ZkProof() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(0)
	return []byte(p.Data()), err
}

func (s Artifact_Approval) HasZkProof() bool {
	return capnp.Struct(s).HasPtr(0)
}

func (s Artifact_Approval) SetZkProof(v []byte) error {
	return capnp.Struct(s).SetData(0, v)
}

func (s Artifact_Approval) OperatorBlindSignature() ([]byte, error) {
	p, err := capnp.Struct(s).Ptr(1)
	return []byte(p.Data()), err
}

func (s Artifact_Approval) HasOperatorBlindSignature() bool {
	return capnp.Struct(s).HasPtr(1)
}

func (s Artifact_Approval) SetOperatorBlindSignature(v []byte) error {
	return capnp.Struct(s).SetData(1, v)
}

// Artifact_Approval_List is a list of Artifact_Approval.
type Artifact_Approval_List = capnp.StructList[Artifact_Approval]

// NewArtifact_Approval creates a new list of Artifact_Approval.
func NewArtifact_Approval_List(s *capnp.Segment, sz int32) (Artifact_Approval_List, error) {
	l, err := capnp.NewCompositeList(s, capnp.ObjectSize{DataSize: 0, PointerCount: 2}, sz)
	return capnp.StructList[Artifact_Approval](l), err
}

// Artifact_Approval_Future is a wrapper for a Artifact_Approval promised by a client call.
type Artifact_Approval_Future struct{ *capnp.Future }

func (f Artifact_Approval_Future) Struct() (Artifact_Approval, error) {
	p, err := f.Future.Ptr()
	return Artifact_Approval(p.Struct()), err
}

const schema_85d3acc39d94e0f8 = "x\xda\x84\xd4K\x88\x1cU\x17\x07\xf0\xff\xffVW?" +
	"2\xdd3ST}\x90/dHb\x02&\x9a\x97\x99" +
	"]\x08\xe4\x81\xc28A\x98\x9a\xd2@BD\xeet\xdf" +
	"\xcc\x94S\xddUTW%\xb6$\xc4\x85\x81 \x11D" +
	"\"\x88\x18p\xa9 \xa2\x1bA\xc1\x85\xbap\x15\x08\xb8" +
	"\x16\\\x180\x0b%\x11!\x11\xa3%\xa7']\xd3\x06" +
	"\xd4\xde\xdd\xdf}\x9d>u\xce\xdd\xbfO\x1d\xa9<\xd1" +
	"\xfaFA\xf9[\xedj\xd1Rw>\xde\xa9\xbe\xba\x02" +
	"g;\xef\xde\xcc?\x9a|\xbc\xf1\x89\xadj\xc0\xec\xd3" +
	"\\\xa2\xfb<k\x80{\x92\xe7P\xce\xfa\x9bh\x15\xf7" +
	"\xbe\xbfz\xed\xeb\x0f\xbf\xbd\x04{R\xd6~\xc1Mt" +
	"\xaf\xf3Q`\xf6\x16\xefZ`\xf1\xb3}\xf0\xfa\x0b\x9f" +
	"n\xfb\x12\xfev\xaa\xbf\x1f|\xb2\xbaD\xb7[\x95\x83" +
	"\xc3\xea9\xb0\x98\xbay\xe2\x97\x037\xbe\xbb\x01\x7f7" +
	"\xd5\xfa\xce\xe7T\x8d60\xfbA\xf5M\x82\xb3\x9fU" +
	"\xb7\x10{\x8aduy_Ggy%\xd5\xfbt\x9a" +
	"\x85gt;\xdb\xdb\xd6I/9xt4<\x9a$" +
	"i\\;\xab\xa3\x05\xd2\xaf[\x15\xa0B\xc0\xd9u\x0c" +
	"\xf0wX\xf4\xf7+:\xa4G\xc1=\x9f\x03\xfe~\x8b" +
	"\xfe!\xc5\x8b/\xaf.\xa4q|\x86-(\xb6\xc0\"" +
	"NL\xaa\xb38\xe5\xb1(\xecu\x82p\xf9pOg" +
	"yj\xca\x05\xa3h\xac\x7f\x8a\xc6jg~\x9dc\xf9" +
	"p\x1a\xf3cIo\xcc\x17\xcf\x98Lwt\xa6\x01\x14" +
	"\xc3\xc0\xcf\xea\x08\x80?7\x8a\xdb\xbd\xc5\xc7\x80\xe0\x07" +
	"Z\x0cnS\xf1A\xe4\xeeO<\x00\x04?\x0a\xffJ" +
	"EG\xd1\xa3\x02\xdc;\x9c\x07\x82\xdb\xe2\xf7\xc5-z" +
	"\xb4\x00\xf77.\x02\xc1=\xf1\x8aRt*\xcac\x05" +
	"p\xa9\x16\x81Ee1h\x0a\xdb\x96'iw\x1b\xea" +
	" \x10T\xc4\xa7\xc5\xab\x15\x8fU\xc0m\x0d\xbd.\xee" +
	"\x89\xd7*\xde\xb0H\x1c%Q6\xc57\x8a\xd7m\x8f" +
	"u\xc0\xfd\x9f\x920\xa7\xc57\x8b7l\x8f\x0d\xc0\xfd" +
	"\xbfJ\x81`\xa3\xf8\x0e\xf1\x0dU\x8f\x1b\x00w\x9b:" +
	"\x05\x04[\xc5w\x8bO\xd4<N\x00\xee.%\x7fk" +
	"\xa7\xf8\x93\xe2\xcd\xba\xc7&\xe0\x1eU\xc7\x80\xe0\x90\xf8" +
	"\x9cx\xab\xe1\xb1\x05\xb8O\xa9\xd7\x80`N\xfcY\xf1" +
	"\xc9\x0d\x1e'\x01\xd7W/\x02\xc1\x82\xf8i\xf1\xa9\x09" +
	"\x8fSR\xe4\xea\x1d 8-\xbe\">\xdd\xf48\x0d" +
	"\xb8F\xd2\x13t\xc4_\x11wZ\x1e\x1d\xc0\xbd0\xf4" +
	"\xf3\xe2\x97\x95\xe2T\x9e\x87\x1d6\xa1\xd8\x04\xb7\xf43" +
	"\x9d\x196\xa0\xd8\x00\x8b\xf6\x8ai\xaf\xf6\xf3.\x80\xb2" +
	"r\xb2\xb0k\xfa\x99\xee\x82\x09m(\xda`\xd15\x9d" +
	"Pg\x83\x044\xa3\x93\x0e\xc7i\xb8\x1c\xf6\xcaa\xd8" +
	"\xef\xe7&\x1d\x0d\xa7\xd282\xacC\xb1.\x97\xb6\xe3" +
	"\xa4\x1c\x15I\xdf\xe4\x9d\xb87\xc0\x96\xee\x9c\xee\xaf\x94" +
	"\x17wM\xba\x1a\x99\xc5\x18V\x9c\x8daY\x84\x9c\x04" +
	"\x17,rz\xbdfA\xc1\x8b\x89\x1eD\xb1\xee\x94\x9b" +
	"L\xaf\x9d\x0e\x92\xcc\xb0\xb3\xb063\xf6\xf7Fs\x98" +
	"\xea\x1c7\x83uNVL\xd7\xa4\x9a\xd1B\xbe\x14\x85" +
	"\xed\xe3\xd6\xd8\xa4\x1eU?\xfb\xebA\x94\xbd\xb2\x16D" +
	"\xd1\x0f\x97\x87=(9z\xb8\x0b\xff\xedM\x18\xb6Y" +
	"Mg\xfa\xa17\xe1\x91\xf57A~\xeb\x8f\x91\xb3\xe7" +
	"\x00Tm\xd5\x0c\xca\x8fzVG\xb9)/\xb3\xff\xeb" +
	"2\x9d\xe9\xbd\xb2\x85\xc6\xf7\xac\xca\xe6\xa2x\xf0\xe2\\" +
	"X\x04\xfc\xf3\x16\xfd\xcb\x8a3\xfc\xb3Xk[\xe7\xd2" +
	"<\xe0\xbfj\xd1\x7fCqF\xfdQ\xacu\xad\xf3\xfa" +
	")\xc0\xbfb\xd1\x7f[q\xc6\xba_\x1c\x196\xad\xf3" +
	"\x96\x1cr\xd5\xa2\xff\x9e\xe2L\xe5wYm\x03\xce\xb5" +
	"%\xc0\x7f\xd7\xa2\xff\xbeb\x91\x99\x97\xb2\x13:\xca\xc7" +
	"\xca\xa9\x08{C2\xf2\xa1Gew&\x8a\xb5(\xac" +
	"\xdcp\x02\x8a\x13`\xb1\x14\xc7\xd1h3\xa1H\xb1\xb0" +
	"\xa7\xd3\xc1\x09\x8dZ\x94\x97\xd9\xff+\x00\x00\xff\xff\x91" +
	">d%"

func RegisterSchema(reg *schemas.Registry) {
	reg.Register(&schemas.Schema{
		String: schema_85d3acc39d94e0f8,
		Nodes: []uint64{
			0x8cc20228b0f1020d,
			0xb1092b0e00ae75e5,
			0xc121b65fcd3a05ee,
			0xd0ddd032f256e50f,
		},
		Compressed: true,
	})
}
