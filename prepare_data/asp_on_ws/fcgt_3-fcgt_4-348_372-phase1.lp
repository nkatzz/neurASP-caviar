
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fcgt_3,164,348). coords(fcgt_3,98,187,348).  visible(fcgt_3,348). happensAt(inactive(fcgt_3),348). orientation(fcgt_4,166,348). coords(fcgt_4,82,206,348).  visible(fcgt_4,348). happensAt(inactive(fcgt_4),348).
orientation(fcgt_3,164,349). coords(fcgt_3,98,187,349).  visible(fcgt_3,349). happensAt(inactive(fcgt_3),349). orientation(fcgt_4,166,349). coords(fcgt_4,82,206,349).  visible(fcgt_4,349). happensAt(inactive(fcgt_4),349).
orientation(fcgt_3,164,350). coords(fcgt_3,98,187,350).  visible(fcgt_3,350). happensAt(inactive(fcgt_3),350). orientation(fcgt_4,166,350). coords(fcgt_4,82,206,350).  visible(fcgt_4,350). happensAt(inactive(fcgt_4),350).
orientation(fcgt_3,164,351). coords(fcgt_3,99,187,351).  visible(fcgt_3,351). happensAt(inactive(fcgt_3),351). orientation(fcgt_4,166,351). coords(fcgt_4,82,206,351).  visible(fcgt_4,351). happensAt(inactive(fcgt_4),351).
orientation(fcgt_3,164,352). coords(fcgt_3,99,187,352).  visible(fcgt_3,352). happensAt(inactive(fcgt_3),352). orientation(fcgt_4,166,352). coords(fcgt_4,82,206,352).  visible(fcgt_4,352). happensAt(inactive(fcgt_4),352).
orientation(fcgt_3,164,353). coords(fcgt_3,99,187,353).  visible(fcgt_3,353). happensAt(inactive(fcgt_3),353). orientation(fcgt_4,166,353). coords(fcgt_4,82,206,353).  visible(fcgt_4,353). happensAt(inactive(fcgt_4),353).
orientation(fcgt_3,164,354). coords(fcgt_3,99,187,354).  visible(fcgt_3,354). happensAt(inactive(fcgt_3),354). orientation(fcgt_4,166,354). coords(fcgt_4,82,206,354).  visible(fcgt_4,354). happensAt(inactive(fcgt_4),354).
orientation(fcgt_3,164,355). coords(fcgt_3,99,187,355).  visible(fcgt_3,355). happensAt(inactive(fcgt_3),355). orientation(fcgt_4,166,355). coords(fcgt_4,82,206,355).  visible(fcgt_4,355). happensAt(inactive(fcgt_4),355).
orientation(fcgt_3,164,356). coords(fcgt_3,99,187,356).  visible(fcgt_3,356). happensAt(inactive(fcgt_3),356). orientation(fcgt_4,166,356). coords(fcgt_4,82,206,356).  visible(fcgt_4,356). happensAt(inactive(fcgt_4),356).
orientation(fcgt_3,164,357). coords(fcgt_3,99,187,357).  visible(fcgt_3,357). happensAt(inactive(fcgt_3),357). orientation(fcgt_4,166,357). coords(fcgt_4,82,206,357).  visible(fcgt_4,357). happensAt(inactive(fcgt_4),357).
orientation(fcgt_3,164,358). coords(fcgt_3,99,187,358).  visible(fcgt_3,358). happensAt(inactive(fcgt_3),358). orientation(fcgt_4,166,358). coords(fcgt_4,82,206,358).  visible(fcgt_4,358). happensAt(inactive(fcgt_4),358).
orientation(fcgt_3,164,359). coords(fcgt_3,99,187,359).  visible(fcgt_3,359). happensAt(inactive(fcgt_3),359). orientation(fcgt_4,166,359). coords(fcgt_4,82,206,359).  visible(fcgt_4,359). happensAt(inactive(fcgt_4),359).
orientation(fcgt_3,164,360). coords(fcgt_3,99,187,360).  visible(fcgt_3,360). happensAt(inactive(fcgt_3),360). orientation(fcgt_4,166,360). coords(fcgt_4,82,206,360).  visible(fcgt_4,360). happensAt(inactive(fcgt_4),360).
orientation(fcgt_3,164,361). coords(fcgt_3,99,187,361).  visible(fcgt_3,361). happensAt(inactive(fcgt_3),361). orientation(fcgt_4,166,361). coords(fcgt_4,82,206,361).  visible(fcgt_4,361). happensAt(inactive(fcgt_4),361).
orientation(fcgt_3,164,362). coords(fcgt_3,99,187,362).  visible(fcgt_3,362). happensAt(inactive(fcgt_3),362). orientation(fcgt_4,166,362). coords(fcgt_4,82,206,362).  visible(fcgt_4,362). happensAt(inactive(fcgt_4),362).
orientation(fcgt_3,164,363). coords(fcgt_3,99,187,363).  visible(fcgt_3,363). happensAt(inactive(fcgt_3),363). orientation(fcgt_4,166,363). coords(fcgt_4,82,206,363).  visible(fcgt_4,363). happensAt(inactive(fcgt_4),363).
orientation(fcgt_3,164,364). coords(fcgt_3,99,187,364).  visible(fcgt_3,364). happensAt(inactive(fcgt_3),364). orientation(fcgt_4,166,364). coords(fcgt_4,82,206,364).  visible(fcgt_4,364). happensAt(inactive(fcgt_4),364).
orientation(fcgt_3,164,365). coords(fcgt_3,99,187,365).  visible(fcgt_3,365). happensAt(inactive(fcgt_3),365). orientation(fcgt_4,166,365). coords(fcgt_4,82,206,365).  visible(fcgt_4,365). happensAt(inactive(fcgt_4),365).
orientation(fcgt_3,164,366). coords(fcgt_3,99,187,366).  visible(fcgt_3,366). happensAt(inactive(fcgt_3),366). orientation(fcgt_4,166,366). coords(fcgt_4,82,206,366).  visible(fcgt_4,366). happensAt(inactive(fcgt_4),366).
orientation(fcgt_3,164,367). coords(fcgt_3,99,187,367).  visible(fcgt_3,367). happensAt(inactive(fcgt_3),367). orientation(fcgt_4,166,367). coords(fcgt_4,82,206,367).  visible(fcgt_4,367). happensAt(inactive(fcgt_4),367).
orientation(fcgt_3,164,368). coords(fcgt_3,99,187,368).  visible(fcgt_3,368). happensAt(inactive(fcgt_3),368). orientation(fcgt_4,166,368). coords(fcgt_4,82,206,368).  visible(fcgt_4,368). happensAt(inactive(fcgt_4),368).
orientation(fcgt_3,164,369). coords(fcgt_3,99,187,369).  visible(fcgt_3,369). happensAt(inactive(fcgt_3),369). orientation(fcgt_4,166,369). coords(fcgt_4,82,206,369).  visible(fcgt_4,369). happensAt(inactive(fcgt_4),369).
orientation(fcgt_3,164,370). coords(fcgt_3,99,187,370).  visible(fcgt_3,370). happensAt(inactive(fcgt_3),370). orientation(fcgt_4,166,370). coords(fcgt_4,82,206,370).  visible(fcgt_4,370). happensAt(inactive(fcgt_4),370).
orientation(fcgt_3,164,371). coords(fcgt_3,99,187,371).  visible(fcgt_3,371). happensAt(inactive(fcgt_3),371). orientation(fcgt_4,166,371). coords(fcgt_4,82,206,371).  visible(fcgt_4,371). happensAt(inactive(fcgt_4),371).
orientation(fcgt_3,164,372). coords(fcgt_3,99,187,372).  visible(fcgt_3,372). happensAt(inactive(fcgt_3),372). orientation(fcgt_4,166,372). coords(fcgt_4,82,206,372).  visible(fcgt_4,372). happensAt(inactive(fcgt_4),372).
#show holdsAt/2.