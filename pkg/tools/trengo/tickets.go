package trengo

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/theapemachine/caramba/pkg/datura"
	"github.com/theapemachine/caramba/pkg/errnie"
)

type Tickets struct {
	client *Client
}

type Ticket struct {
	ID         int    `json:"id"`
	Subject    string `json:"subject"`
	Body       string `json:"body"`
	Status     string `json:"status"`
	AssignedTo int    `json:"assigned_to"`
	ChannelID  int    `json:"channel_id"`
	ContactID  int    `json:"contact_id"`
	CreatedAt  string `json:"created_at"`
	UpdatedAt  string `json:"updated_at"`
}

type TicketList struct {
	Data []Ticket `json:"data"`
}

func NewTickets(client *Client) *Tickets {
	return &Tickets{client: client}
}

func (t *Tickets) encode(artifact *datura.Artifact, v any) (err error) {
	data, err := json.Marshal(v)
	if err != nil {
		return errnie.Error(err)
	}

	datura.WithPayload(data)(artifact)
	return nil
}

func (t *Tickets) ListTickets(artifact *datura.Artifact) (err error) {
	resp, err := t.client.doRequest(http.MethodGet, "/tickets", nil)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return errnie.Error(err)
	}

	var tickets TicketList
	if err = json.Unmarshal(body, &tickets); err != nil {
		return errnie.Error(err)
	}

	return t.encode(artifact, tickets)
}

func (t *Tickets) CreateTicket(artifact *datura.Artifact) (err error) {
	ticket := map[string]interface{}{
		"subject":    datura.GetMetaValue[string](artifact, "subject"),
		"body":       datura.GetMetaValue[string](artifact, "body"),
		"channel_id": datura.GetMetaValue[int](artifact, "channel_id"),
		"contact_id": datura.GetMetaValue[int](artifact, "contact_id"),
	}

	resp, err := t.client.doRequest(http.MethodPost, "/tickets", ticket)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	var newTicket Ticket
	if err = json.NewDecoder(resp.Body).Decode(&newTicket); err != nil {
		return errnie.Error(err)
	}

	return t.encode(artifact, newTicket)
}

func (t *Tickets) AssignTicket(artifact *datura.Artifact) (err error) {
	ticketID := datura.GetMetaValue[int](artifact, "ticket_id")
	userID := datura.GetMetaValue[int](artifact, "user_id")

	resp, err := t.client.doRequest(
		http.MethodPut,
		fmt.Sprintf("/tickets/%d/assign/%d", ticketID, userID),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	var ticket Ticket
	if err = json.NewDecoder(resp.Body).Decode(&ticket); err != nil {
		return errnie.Error(err)
	}

	return t.encode(artifact, ticket)
}

func (t *Tickets) CloseTicket(artifact *datura.Artifact) (err error) {
	ticketID := datura.GetMetaValue[int](artifact, "ticket_id")

	resp, err := t.client.doRequest(
		http.MethodPut,
		fmt.Sprintf("/tickets/%d/close", ticketID),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	var ticket Ticket
	if err = json.NewDecoder(resp.Body).Decode(&ticket); err != nil {
		return errnie.Error(err)
	}

	return t.encode(artifact, ticket)
}

func (t *Tickets) ReopenTicket(artifact *datura.Artifact) (err error) {
	ticketID := datura.GetMetaValue[int](artifact, "ticket_id")

	resp, err := t.client.doRequest(
		http.MethodPut,
		fmt.Sprintf("/tickets/%d/reopen", ticketID),
		nil,
	)
	if err != nil {
		return errnie.Error(err)
	}
	defer resp.Body.Close()

	var ticket Ticket
	if err = json.NewDecoder(resp.Body).Decode(&ticket); err != nil {
		return errnie.Error(err)
	}

	return t.encode(artifact, ticket)
}
