
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

	orientation(fcgt_1,141,352). coords(fcgt_1,96,123,352).  visible(fcgt_1,352). happensAt(active(fcgt_1),352). orientation(fcgt_3,164,352). coords(fcgt_3,99,187,352).  visible(fcgt_3,352). happensAt(inactive(fcgt_3),352).
orientation(fcgt_1,141,353). coords(fcgt_1,96,121,353).  visible(fcgt_1,353). happensAt(active(fcgt_1),353). orientation(fcgt_3,164,353). coords(fcgt_3,99,187,353).  visible(fcgt_3,353). happensAt(inactive(fcgt_3),353).
orientation(fcgt_1,141,354). coords(fcgt_1,96,121,354).  visible(fcgt_1,354). happensAt(active(fcgt_1),354). orientation(fcgt_3,164,354). coords(fcgt_3,99,187,354).  visible(fcgt_3,354). happensAt(inactive(fcgt_3),354).
orientation(fcgt_1,141,355). coords(fcgt_1,96,121,355).  visible(fcgt_1,355). happensAt(active(fcgt_1),355). orientation(fcgt_3,164,355). coords(fcgt_3,99,187,355).  visible(fcgt_3,355). happensAt(inactive(fcgt_3),355).
orientation(fcgt_1,141,356). coords(fcgt_1,96,120,356).  visible(fcgt_1,356). happensAt(active(fcgt_1),356). orientation(fcgt_3,164,356). coords(fcgt_3,99,187,356).  visible(fcgt_3,356). happensAt(inactive(fcgt_3),356).
orientation(fcgt_1,141,357). coords(fcgt_1,96,121,357).  visible(fcgt_1,357). happensAt(active(fcgt_1),357). orientation(fcgt_3,164,357). coords(fcgt_3,99,187,357).  visible(fcgt_3,357). happensAt(inactive(fcgt_3),357).
orientation(fcgt_1,141,358). coords(fcgt_1,96,121,358).  visible(fcgt_1,358). happensAt(active(fcgt_1),358). orientation(fcgt_3,164,358). coords(fcgt_3,99,187,358).  visible(fcgt_3,358). happensAt(inactive(fcgt_3),358).
orientation(fcgt_1,141,359). coords(fcgt_1,96,122,359).  visible(fcgt_1,359). happensAt(active(fcgt_1),359). orientation(fcgt_3,164,359). coords(fcgt_3,99,187,359).  visible(fcgt_3,359). happensAt(inactive(fcgt_3),359).
orientation(fcgt_1,141,360). coords(fcgt_1,95,122,360).  visible(fcgt_1,360). happensAt(active(fcgt_1),360). orientation(fcgt_3,164,360). coords(fcgt_3,99,187,360).  visible(fcgt_3,360). happensAt(inactive(fcgt_3),360).
orientation(fcgt_1,141,361). coords(fcgt_1,95,121,361).  visible(fcgt_1,361). happensAt(active(fcgt_1),361). orientation(fcgt_3,164,361). coords(fcgt_3,99,187,361).  visible(fcgt_3,361). happensAt(inactive(fcgt_3),361).
orientation(fcgt_1,141,362). coords(fcgt_1,95,120,362).  visible(fcgt_1,362). happensAt(active(fcgt_1),362). orientation(fcgt_3,164,362). coords(fcgt_3,99,187,362).  visible(fcgt_3,362). happensAt(inactive(fcgt_3),362).
orientation(fcgt_1,141,363). coords(fcgt_1,95,117,363).  visible(fcgt_1,363). happensAt(active(fcgt_1),363). orientation(fcgt_3,164,363). coords(fcgt_3,99,187,363).  visible(fcgt_3,363). happensAt(inactive(fcgt_3),363).
orientation(fcgt_1,141,364). coords(fcgt_1,95,115,364).  visible(fcgt_1,364). happensAt(active(fcgt_1),364). orientation(fcgt_3,164,364). coords(fcgt_3,99,187,364).  visible(fcgt_3,364). happensAt(inactive(fcgt_3),364).
orientation(fcgt_1,141,365). coords(fcgt_1,95,112,365).  visible(fcgt_1,365). happensAt(active(fcgt_1),365). orientation(fcgt_3,164,365). coords(fcgt_3,99,187,365).  visible(fcgt_3,365). happensAt(inactive(fcgt_3),365).
orientation(fcgt_1,141,366). coords(fcgt_1,94,109,366).  visible(fcgt_1,366). happensAt(active(fcgt_1),366). orientation(fcgt_3,164,366). coords(fcgt_3,99,187,366).  visible(fcgt_3,366). happensAt(inactive(fcgt_3),366).
orientation(fcgt_1,141,367). coords(fcgt_1,94,107,367).  visible(fcgt_1,367). happensAt(running(fcgt_1),367). orientation(fcgt_3,164,367). coords(fcgt_3,99,187,367).  visible(fcgt_3,367). happensAt(inactive(fcgt_3),367).
orientation(fcgt_1,141,368). coords(fcgt_1,92,105,368).  visible(fcgt_1,368). happensAt(running(fcgt_1),368). orientation(fcgt_3,164,368). coords(fcgt_3,99,187,368).  visible(fcgt_3,368). happensAt(inactive(fcgt_3),368).
orientation(fcgt_1,141,369). coords(fcgt_1,90,104,369).  visible(fcgt_1,369). happensAt(running(fcgt_1),369). orientation(fcgt_3,164,369). coords(fcgt_3,99,187,369).  visible(fcgt_3,369). happensAt(inactive(fcgt_3),369).
orientation(fcgt_1,141,370). coords(fcgt_1,88,103,370).  visible(fcgt_1,370). happensAt(running(fcgt_1),370). orientation(fcgt_3,164,370). coords(fcgt_3,99,187,370).  visible(fcgt_3,370). happensAt(inactive(fcgt_3),370).
orientation(fcgt_1,141,371). coords(fcgt_1,87,103,371).  visible(fcgt_1,371). happensAt(running(fcgt_1),371). orientation(fcgt_3,164,371). coords(fcgt_3,99,187,371).  visible(fcgt_3,371). happensAt(inactive(fcgt_3),371).
orientation(fcgt_1,141,372). coords(fcgt_1,85,102,372).  visible(fcgt_1,372). happensAt(running(fcgt_1),372). orientation(fcgt_3,164,372). coords(fcgt_3,99,187,372).  visible(fcgt_3,372). happensAt(inactive(fcgt_3),372).
orientation(fcgt_1,141,373). coords(fcgt_1,84,102,373).  visible(fcgt_1,373). happensAt(running(fcgt_1),373). orientation(fcgt_3,164,373). coords(fcgt_3,99,187,373).  visible(fcgt_3,373). happensAt(inactive(fcgt_3),373).
orientation(fcgt_1,149,374). coords(fcgt_1,83,102,374).  visible(fcgt_1,374). happensAt(running(fcgt_1),374). orientation(fcgt_3,164,374). coords(fcgt_3,99,187,374).  visible(fcgt_3,374). happensAt(inactive(fcgt_3),374).
orientation(fcgt_1,150,375). coords(fcgt_1,83,101,375).  visible(fcgt_1,375). happensAt(running(fcgt_1),375). orientation(fcgt_3,164,375). coords(fcgt_3,99,187,375).  visible(fcgt_3,375). happensAt(inactive(fcgt_3),375).
orientation(fcgt_1,150,376). coords(fcgt_1,83,100,376).  visible(fcgt_1,376). happensAt(running(fcgt_1),376). orientation(fcgt_3,164,376). coords(fcgt_3,99,187,376).  visible(fcgt_3,376). happensAt(inactive(fcgt_3),376).
#show holdsAt/2.