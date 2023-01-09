
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
orientation(fcgt_1,155,377). coords(fcgt_1,83,98,377).  visible(fcgt_1,377). happensAt(running(fcgt_1),377). orientation(fcgt_3,164,377). coords(fcgt_3,99,187,377).  visible(fcgt_3,377). happensAt(inactive(fcgt_3),377).
orientation(fcgt_1,154,378). coords(fcgt_1,82,98,378).  visible(fcgt_1,378). happensAt(running(fcgt_1),378). orientation(fcgt_3,164,378). coords(fcgt_3,99,187,378).  visible(fcgt_3,378). happensAt(inactive(fcgt_3),378).
orientation(fcgt_1,155,379). coords(fcgt_1,82,97,379).  visible(fcgt_1,379). happensAt(running(fcgt_1),379). orientation(fcgt_3,164,379). coords(fcgt_3,99,187,379).  visible(fcgt_3,379). happensAt(walking(fcgt_3),379).
orientation(fcgt_1,157,380). coords(fcgt_1,82,97,380).  visible(fcgt_1,380). happensAt(running(fcgt_1),380). orientation(fcgt_3,164,380). coords(fcgt_3,99,187,380).  visible(fcgt_3,380). happensAt(walking(fcgt_3),380).
orientation(fcgt_1,157,381). coords(fcgt_1,82,96,381).  visible(fcgt_1,381). happensAt(running(fcgt_1),381). orientation(fcgt_3,164,381). coords(fcgt_3,99,187,381).  visible(fcgt_3,381). happensAt(walking(fcgt_3),381).
orientation(fcgt_1,157,382). coords(fcgt_1,81,96,382).  visible(fcgt_1,382). happensAt(running(fcgt_1),382). orientation(fcgt_3,164,382). coords(fcgt_3,99,187,382).  visible(fcgt_3,382). happensAt(walking(fcgt_3),382).
orientation(fcgt_1,157,383). coords(fcgt_1,80,96,383).  visible(fcgt_1,383). happensAt(running(fcgt_1),383). orientation(fcgt_3,164,383). coords(fcgt_3,99,187,383).  visible(fcgt_3,383). happensAt(walking(fcgt_3),383).
orientation(fcgt_1,157,384). coords(fcgt_1,79,96,384).  visible(fcgt_1,384). happensAt(running(fcgt_1),384). orientation(fcgt_3,164,384). coords(fcgt_3,99,187,384).  visible(fcgt_3,384). happensAt(walking(fcgt_3),384).
orientation(fcgt_1,157,385). coords(fcgt_1,78,96,385).  visible(fcgt_1,385). happensAt(running(fcgt_1),385). orientation(fcgt_3,164,385). coords(fcgt_3,99,188,385).  visible(fcgt_3,385). happensAt(walking(fcgt_3),385).
orientation(fcgt_1,152,386). coords(fcgt_1,78,96,386).  visible(fcgt_1,386). happensAt(running(fcgt_1),386). orientation(fcgt_3,164,386). coords(fcgt_3,99,188,386).  visible(fcgt_3,386). happensAt(walking(fcgt_3),386).
orientation(fcgt_1,146,387). coords(fcgt_1,77,96,387).  visible(fcgt_1,387). happensAt(running(fcgt_1),387). orientation(fcgt_3,164,387). coords(fcgt_3,99,188,387).  visible(fcgt_3,387). happensAt(walking(fcgt_3),387).
orientation(fcgt_1,146,388). coords(fcgt_1,77,96,388).  visible(fcgt_1,388). happensAt(running(fcgt_1),388). orientation(fcgt_3,164,388). coords(fcgt_3,99,189,388).  visible(fcgt_3,388). happensAt(walking(fcgt_3),388).
#show holdsAt/2.