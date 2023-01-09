
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

	orientation(fcgt_2,146,354). coords(fcgt_2,92,110,354).  visible(fcgt_2,354). happensAt(active(fcgt_2),354). orientation(fcgt_4,166,354). coords(fcgt_4,82,206,354).  visible(fcgt_4,354). happensAt(inactive(fcgt_4),354).
orientation(fcgt_2,147,355). coords(fcgt_2,92,109,355).  visible(fcgt_2,355). happensAt(active(fcgt_2),355). orientation(fcgt_4,166,355). coords(fcgt_4,82,206,355).  visible(fcgt_4,355). happensAt(inactive(fcgt_4),355).
orientation(fcgt_2,142,356). coords(fcgt_2,92,107,356).  visible(fcgt_2,356). happensAt(active(fcgt_2),356). orientation(fcgt_4,166,356). coords(fcgt_4,82,206,356).  visible(fcgt_4,356). happensAt(inactive(fcgt_4),356).
orientation(fcgt_2,142,357). coords(fcgt_2,92,106,357).  visible(fcgt_2,357). happensAt(active(fcgt_2),357). orientation(fcgt_4,166,357). coords(fcgt_4,82,206,357).  visible(fcgt_4,357). happensAt(inactive(fcgt_4),357).
orientation(fcgt_2,142,358). coords(fcgt_2,92,104,358).  visible(fcgt_2,358). happensAt(active(fcgt_2),358). orientation(fcgt_4,166,358). coords(fcgt_4,82,206,358).  visible(fcgt_4,358). happensAt(inactive(fcgt_4),358).
orientation(fcgt_2,142,359). coords(fcgt_2,91,104,359).  visible(fcgt_2,359). happensAt(active(fcgt_2),359). orientation(fcgt_4,166,359). coords(fcgt_4,82,206,359).  visible(fcgt_4,359). happensAt(inactive(fcgt_4),359).
orientation(fcgt_2,142,360). coords(fcgt_2,91,103,360).  visible(fcgt_2,360). happensAt(active(fcgt_2),360). orientation(fcgt_4,166,360). coords(fcgt_4,82,206,360).  visible(fcgt_4,360). happensAt(inactive(fcgt_4),360).
orientation(fcgt_2,142,361). coords(fcgt_2,91,102,361).  visible(fcgt_2,361). happensAt(active(fcgt_2),361). orientation(fcgt_4,166,361). coords(fcgt_4,82,206,361).  visible(fcgt_4,361). happensAt(inactive(fcgt_4),361).
orientation(fcgt_2,142,362). coords(fcgt_2,91,102,362).  visible(fcgt_2,362). happensAt(active(fcgt_2),362). orientation(fcgt_4,166,362). coords(fcgt_4,82,206,362).  visible(fcgt_4,362). happensAt(inactive(fcgt_4),362).
orientation(fcgt_2,142,363). coords(fcgt_2,91,102,363).  visible(fcgt_2,363). happensAt(active(fcgt_2),363). orientation(fcgt_4,166,363). coords(fcgt_4,82,206,363).  visible(fcgt_4,363). happensAt(inactive(fcgt_4),363).
orientation(fcgt_2,142,364). coords(fcgt_2,91,102,364).  visible(fcgt_2,364). happensAt(active(fcgt_2),364). orientation(fcgt_4,166,364). coords(fcgt_4,82,206,364).  visible(fcgt_4,364). happensAt(inactive(fcgt_4),364).
orientation(fcgt_2,142,365). coords(fcgt_2,90,101,365).  visible(fcgt_2,365). happensAt(active(fcgt_2),365). orientation(fcgt_4,166,365). coords(fcgt_4,82,206,365).  visible(fcgt_4,365). happensAt(inactive(fcgt_4),365).
orientation(fcgt_2,142,366). coords(fcgt_2,90,101,366).  visible(fcgt_2,366). happensAt(active(fcgt_2),366). orientation(fcgt_4,166,366). coords(fcgt_4,82,206,366).  visible(fcgt_4,366). happensAt(inactive(fcgt_4),366).
orientation(fcgt_2,142,367). coords(fcgt_2,90,101,367).  visible(fcgt_2,367). happensAt(running(fcgt_2),367). orientation(fcgt_4,166,367). coords(fcgt_4,82,206,367).  visible(fcgt_4,367). happensAt(inactive(fcgt_4),367).
orientation(fcgt_2,142,368). coords(fcgt_2,90,101,368).  visible(fcgt_2,368). happensAt(running(fcgt_2),368). orientation(fcgt_4,166,368). coords(fcgt_4,82,206,368).  visible(fcgt_4,368). happensAt(inactive(fcgt_4),368).
orientation(fcgt_2,142,369). coords(fcgt_2,90,100,369).  visible(fcgt_2,369). happensAt(running(fcgt_2),369). orientation(fcgt_4,166,369). coords(fcgt_4,82,206,369).  visible(fcgt_4,369). happensAt(inactive(fcgt_4),369).
orientation(fcgt_2,142,370). coords(fcgt_2,88,100,370).  visible(fcgt_2,370). happensAt(running(fcgt_2),370). orientation(fcgt_4,166,370). coords(fcgt_4,82,206,370).  visible(fcgt_4,370). happensAt(inactive(fcgt_4),370).
orientation(fcgt_2,142,371). coords(fcgt_2,88,100,371).  visible(fcgt_2,371). happensAt(running(fcgt_2),371). orientation(fcgt_4,166,371). coords(fcgt_4,82,206,371).  visible(fcgt_4,371). happensAt(inactive(fcgt_4),371).
orientation(fcgt_2,142,372). coords(fcgt_2,87,97,372).  visible(fcgt_2,372). happensAt(running(fcgt_2),372). orientation(fcgt_4,166,372). coords(fcgt_4,82,206,372).  visible(fcgt_4,372). happensAt(inactive(fcgt_4),372).
orientation(fcgt_2,142,373). coords(fcgt_2,87,96,373).  visible(fcgt_2,373). happensAt(running(fcgt_2),373). orientation(fcgt_4,166,373). coords(fcgt_4,82,206,373).  visible(fcgt_4,373). happensAt(inactive(fcgt_4),373).
orientation(fcgt_2,142,374). coords(fcgt_2,86,95,374).  visible(fcgt_2,374). happensAt(running(fcgt_2),374). orientation(fcgt_4,166,374). coords(fcgt_4,82,206,374).  visible(fcgt_4,374). happensAt(inactive(fcgt_4),374).
orientation(fcgt_2,142,375). coords(fcgt_2,84,95,375).  visible(fcgt_2,375). happensAt(running(fcgt_2),375). orientation(fcgt_4,166,375). coords(fcgt_4,82,206,375).  visible(fcgt_4,375). happensAt(inactive(fcgt_4),375).
orientation(fcgt_2,142,376). coords(fcgt_2,84,94,376).  visible(fcgt_2,376). happensAt(running(fcgt_2),376). orientation(fcgt_4,166,376). coords(fcgt_4,82,206,376).  visible(fcgt_4,376). happensAt(inactive(fcgt_4),376).
orientation(fcgt_2,142,377). coords(fcgt_2,84,93,377).  visible(fcgt_2,377). happensAt(running(fcgt_2),377). orientation(fcgt_4,166,377). coords(fcgt_4,82,206,377).  visible(fcgt_4,377). happensAt(inactive(fcgt_4),377).
orientation(fcgt_2,142,378). coords(fcgt_2,84,93,378).  visible(fcgt_2,378). happensAt(running(fcgt_2),378). orientation(fcgt_4,166,378). coords(fcgt_4,82,206,378).  visible(fcgt_4,378). happensAt(inactive(fcgt_4),378).
#show holdsAt/2.