
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

	orientation(fcgt_1,141,352). coords(fcgt_1,96,123,352).  visible(fcgt_1,352). happensAt(active(fcgt_1),352). orientation(fcgt_2,147,352). coords(fcgt_2,92,111,352).  visible(fcgt_2,352). happensAt(active(fcgt_2),352).
orientation(fcgt_1,141,353). coords(fcgt_1,96,121,353).  visible(fcgt_1,353). happensAt(active(fcgt_1),353). orientation(fcgt_2,148,353). coords(fcgt_2,92,110,353).  visible(fcgt_2,353). happensAt(active(fcgt_2),353).
orientation(fcgt_1,141,354). coords(fcgt_1,96,121,354).  visible(fcgt_1,354). happensAt(active(fcgt_1),354). orientation(fcgt_2,146,354). coords(fcgt_2,92,110,354).  visible(fcgt_2,354). happensAt(active(fcgt_2),354).
orientation(fcgt_1,141,355). coords(fcgt_1,96,121,355).  visible(fcgt_1,355). happensAt(active(fcgt_1),355). orientation(fcgt_2,147,355). coords(fcgt_2,92,109,355).  visible(fcgt_2,355). happensAt(active(fcgt_2),355).
orientation(fcgt_1,141,356). coords(fcgt_1,96,120,356).  visible(fcgt_1,356). happensAt(active(fcgt_1),356). orientation(fcgt_2,142,356). coords(fcgt_2,92,107,356).  visible(fcgt_2,356). happensAt(active(fcgt_2),356).
orientation(fcgt_1,141,357). coords(fcgt_1,96,121,357).  visible(fcgt_1,357). happensAt(active(fcgt_1),357). orientation(fcgt_2,142,357). coords(fcgt_2,92,106,357).  visible(fcgt_2,357). happensAt(active(fcgt_2),357).
orientation(fcgt_1,141,358). coords(fcgt_1,96,121,358).  visible(fcgt_1,358). happensAt(active(fcgt_1),358). orientation(fcgt_2,142,358). coords(fcgt_2,92,104,358).  visible(fcgt_2,358). happensAt(active(fcgt_2),358).
orientation(fcgt_1,141,359). coords(fcgt_1,96,122,359).  visible(fcgt_1,359). happensAt(active(fcgt_1),359). orientation(fcgt_2,142,359). coords(fcgt_2,91,104,359).  visible(fcgt_2,359). happensAt(active(fcgt_2),359).
orientation(fcgt_1,141,360). coords(fcgt_1,95,122,360).  visible(fcgt_1,360). happensAt(active(fcgt_1),360). orientation(fcgt_2,142,360). coords(fcgt_2,91,103,360).  visible(fcgt_2,360). happensAt(active(fcgt_2),360).
orientation(fcgt_1,141,361). coords(fcgt_1,95,121,361).  visible(fcgt_1,361). happensAt(active(fcgt_1),361). orientation(fcgt_2,142,361). coords(fcgt_2,91,102,361).  visible(fcgt_2,361). happensAt(active(fcgt_2),361).
orientation(fcgt_1,141,362). coords(fcgt_1,95,120,362).  visible(fcgt_1,362). happensAt(active(fcgt_1),362). orientation(fcgt_2,142,362). coords(fcgt_2,91,102,362).  visible(fcgt_2,362). happensAt(active(fcgt_2),362).
orientation(fcgt_1,141,363). coords(fcgt_1,95,117,363).  visible(fcgt_1,363). happensAt(active(fcgt_1),363). orientation(fcgt_2,142,363). coords(fcgt_2,91,102,363).  visible(fcgt_2,363). happensAt(active(fcgt_2),363).
orientation(fcgt_1,141,364). coords(fcgt_1,95,115,364).  visible(fcgt_1,364). happensAt(active(fcgt_1),364). orientation(fcgt_2,142,364). coords(fcgt_2,91,102,364).  visible(fcgt_2,364). happensAt(active(fcgt_2),364).
orientation(fcgt_1,141,365). coords(fcgt_1,95,112,365).  visible(fcgt_1,365). happensAt(active(fcgt_1),365). orientation(fcgt_2,142,365). coords(fcgt_2,90,101,365).  visible(fcgt_2,365). happensAt(active(fcgt_2),365).
orientation(fcgt_1,141,366). coords(fcgt_1,94,109,366).  visible(fcgt_1,366). happensAt(active(fcgt_1),366). orientation(fcgt_2,142,366). coords(fcgt_2,90,101,366).  visible(fcgt_2,366). happensAt(active(fcgt_2),366).
orientation(fcgt_1,141,367). coords(fcgt_1,94,107,367).  visible(fcgt_1,367). happensAt(running(fcgt_1),367). orientation(fcgt_2,142,367). coords(fcgt_2,90,101,367).  visible(fcgt_2,367). happensAt(running(fcgt_2),367).
orientation(fcgt_1,141,368). coords(fcgt_1,92,105,368).  visible(fcgt_1,368). happensAt(running(fcgt_1),368). orientation(fcgt_2,142,368). coords(fcgt_2,90,101,368).  visible(fcgt_2,368). happensAt(running(fcgt_2),368).
orientation(fcgt_1,141,369). coords(fcgt_1,90,104,369).  visible(fcgt_1,369). happensAt(running(fcgt_1),369). orientation(fcgt_2,142,369). coords(fcgt_2,90,100,369).  visible(fcgt_2,369). happensAt(running(fcgt_2),369).
orientation(fcgt_1,141,370). coords(fcgt_1,88,103,370).  visible(fcgt_1,370). happensAt(running(fcgt_1),370). orientation(fcgt_2,142,370). coords(fcgt_2,88,100,370).  visible(fcgt_2,370). happensAt(running(fcgt_2),370).
orientation(fcgt_1,141,371). coords(fcgt_1,87,103,371).  visible(fcgt_1,371). happensAt(running(fcgt_1),371). orientation(fcgt_2,142,371). coords(fcgt_2,88,100,371).  visible(fcgt_2,371). happensAt(running(fcgt_2),371).
orientation(fcgt_1,141,372). coords(fcgt_1,85,102,372).  visible(fcgt_1,372). happensAt(running(fcgt_1),372). orientation(fcgt_2,142,372). coords(fcgt_2,87,97,372).  visible(fcgt_2,372). happensAt(running(fcgt_2),372).
orientation(fcgt_1,141,373). coords(fcgt_1,84,102,373).  visible(fcgt_1,373). happensAt(running(fcgt_1),373). orientation(fcgt_2,142,373). coords(fcgt_2,87,96,373).  visible(fcgt_2,373). happensAt(running(fcgt_2),373).
orientation(fcgt_1,149,374). coords(fcgt_1,83,102,374).  visible(fcgt_1,374). happensAt(running(fcgt_1),374). orientation(fcgt_2,142,374). coords(fcgt_2,86,95,374).  visible(fcgt_2,374). happensAt(running(fcgt_2),374).
orientation(fcgt_1,150,375). coords(fcgt_1,83,101,375).  visible(fcgt_1,375). happensAt(running(fcgt_1),375). orientation(fcgt_2,142,375). coords(fcgt_2,84,95,375).  visible(fcgt_2,375). happensAt(running(fcgt_2),375).
orientation(fcgt_1,150,376). coords(fcgt_1,83,100,376).  visible(fcgt_1,376). happensAt(running(fcgt_1),376). orientation(fcgt_2,142,376). coords(fcgt_2,84,94,376).  visible(fcgt_2,376). happensAt(running(fcgt_2),376).
#show holdsAt/2.