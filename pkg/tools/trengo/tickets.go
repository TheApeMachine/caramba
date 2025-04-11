package trengo

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"

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

func (t *Tickets) ListTickets() (tickets TicketList, err error) {
	resp, err := t.client.doRequest(http.MethodGet, "/tickets", nil)
	if err != nil {
		return TicketList{}, errnie.Error(err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return TicketList{}, errnie.Error(err)
	}

	if err = json.Unmarshal(body, &tickets); err != nil {
		return TicketList{}, errnie.Error(err)
	}

	return tickets, nil
}

func (t *Tickets) CreateTicket(ticket Ticket) (newTicket Ticket, err error) {
	resp, err := t.client.doRequest(http.MethodPost, "/tickets", ticket)
	if err != nil {
		return Ticket{}, errnie.Error(err)
	}
	defer resp.Body.Close()

	if err = json.NewDecoder(resp.Body).Decode(&newTicket); err != nil {
		return Ticket{}, errnie.Error(err)
	}

	return newTicket, nil
}

func (t *Tickets) AssignTicket(ticketID int, userID int) (err error) {
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

	return nil
}

func (t *Tickets) CloseTicket(ticketID int) (err error) {
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

	return nil
}

func (t *Tickets) ReopenTicket(ticketID int) (err error) {
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

	return nil
}
