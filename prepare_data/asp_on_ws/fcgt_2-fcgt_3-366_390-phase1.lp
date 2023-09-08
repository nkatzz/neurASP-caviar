
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

	orientation(fcgt_2,142,366). coords(fcgt_2,90,101,366).  visible(fcgt_2,366). happensAt(active(fcgt_2),366). orientation(fcgt_3,164,366). coords(fcgt_3,99,187,366).  visible(fcgt_3,366). happensAt(inactive(fcgt_3),366).
orientation(fcgt_2,142,367). coords(fcgt_2,90,101,367).  visible(fcgt_2,367). happensAt(running(fcgt_2),367). orientation(fcgt_3,164,367). coords(fcgt_3,99,187,367).  visible(fcgt_3,367). happensAt(inactive(fcgt_3),367).
orientation(fcgt_2,142,368). coords(fcgt_2,90,101,368).  visible(fcgt_2,368). happensAt(running(fcgt_2),368). orientation(fcgt_3,164,368). coords(fcgt_3,99,187,368).  visible(fcgt_3,368). happensAt(inactive(fcgt_3),368).
orientation(fcgt_2,142,369). coords(fcgt_2,90,100,369).  visible(fcgt_2,369). happensAt(running(fcgt_2),369). orientation(fcgt_3,164,369). coords(fcgt_3,99,187,369).  visible(fcgt_3,369). happensAt(inactive(fcgt_3),369).
orientation(fcgt_2,142,370). coords(fcgt_2,88,100,370).  visible(fcgt_2,370). happensAt(running(fcgt_2),370). orientation(fcgt_3,164,370). coords(fcgt_3,99,187,370).  visible(fcgt_3,370). happensAt(inactive(fcgt_3),370).
orientation(fcgt_2,142,371). coords(fcgt_2,88,100,371).  visible(fcgt_2,371). happensAt(running(fcgt_2),371). orientation(fcgt_3,164,371). coords(fcgt_3,99,187,371).  visible(fcgt_3,371). happensAt(inactive(fcgt_3),371).
orientation(fcgt_2,142,372). coords(fcgt_2,87,97,372).  visible(fcgt_2,372). happensAt(running(fcgt_2),372). orientation(fcgt_3,164,372). coords(fcgt_3,99,187,372).  visible(fcgt_3,372). happensAt(inactive(fcgt_3),372).
orientation(fcgt_2,142,373). coords(fcgt_2,87,96,373).  visible(fcgt_2,373). happensAt(running(fcgt_2),373). orientation(fcgt_3,164,373). coords(fcgt_3,99,187,373).  visible(fcgt_3,373). happensAt(inactive(fcgt_3),373).
orientation(fcgt_2,142,374). coords(fcgt_2,86,95,374).  visible(fcgt_2,374). happensAt(running(fcgt_2),374). orientation(fcgt_3,164,374). coords(fcgt_3,99,187,374).  visible(fcgt_3,374). happensAt(inactive(fcgt_3),374).
orientation(fcgt_2,142,375). coords(fcgt_2,84,95,375).  visible(fcgt_2,375). happensAt(running(fcgt_2),375). orientation(fcgt_3,164,375). coords(fcgt_3,99,187,375).  visible(fcgt_3,375). happensAt(inactive(fcgt_3),375).
orientation(fcgt_2,142,376). coords(fcgt_2,84,94,376).  visible(fcgt_2,376). happensAt(running(fcgt_2),376). orientation(fcgt_3,164,376). coords(fcgt_3,99,187,376).  visible(fcgt_3,376). happensAt(inactive(fcgt_3),376).
orientation(fcgt_2,142,377). coords(fcgt_2,84,93,377).  visible(fcgt_2,377). happensAt(running(fcgt_2),377). orientation(fcgt_3,164,377). coords(fcgt_3,99,187,377).  visible(fcgt_3,377). happensAt(inactive(fcgt_3),377).
orientation(fcgt_2,142,378). coords(fcgt_2,84,93,378).  visible(fcgt_2,378). happensAt(running(fcgt_2),378). orientation(fcgt_3,164,378). coords(fcgt_3,99,187,378).  visible(fcgt_3,378). happensAt(inactive(fcgt_3),378).
orientation(fcgt_2,142,379). coords(fcgt_2,84,93,379).  visible(fcgt_2,379). happensAt(running(fcgt_2),379). orientation(fcgt_3,164,379). coords(fcgt_3,99,187,379).  visible(fcgt_3,379). happensAt(walking(fcgt_3),379).
orientation(fcgt_2,142,380). coords(fcgt_2,84,93,380).  visible(fcgt_2,380). happensAt(running(fcgt_2),380). orientation(fcgt_3,164,380). coords(fcgt_3,99,187,380).  visible(fcgt_3,380). happensAt(walking(fcgt_3),380).
orientation(fcgt_2,142,381). coords(fcgt_2,84,93,381).  visible(fcgt_2,381). happensAt(running(fcgt_2),381). orientation(fcgt_3,164,381). coords(fcgt_3,99,187,381).  visible(fcgt_3,381). happensAt(walking(fcgt_3),381).
orientation(fcgt_2,142,382). coords(fcgt_2,83,92,382).  visible(fcgt_2,382). happensAt(running(fcgt_2),382). orientation(fcgt_3,164,382). coords(fcgt_3,99,187,382).  visible(fcgt_3,382). happensAt(walking(fcgt_3),382).
orientation(fcgt_2,142,383). coords(fcgt_2,83,92,383).  visible(fcgt_2,383). happensAt(running(fcgt_2),383). orientation(fcgt_3,164,383). coords(fcgt_3,99,187,383).  visible(fcgt_3,383). happensAt(walking(fcgt_3),383).
orientation(fcgt_2,142,384). coords(fcgt_2,82,91,384).  visible(fcgt_2,384). happensAt(running(fcgt_2),384). orientation(fcgt_3,164,384). coords(fcgt_3,99,187,384).  visible(fcgt_3,384). happensAt(walking(fcgt_3),384).
orientation(fcgt_2,142,385). coords(fcgt_2,81,90,385).  visible(fcgt_2,385). happensAt(running(fcgt_2),385). orientation(fcgt_3,164,385). coords(fcgt_3,99,188,385).  visible(fcgt_3,385). happensAt(walking(fcgt_3),385).
orientation(fcgt_2,142,386). coords(fcgt_2,81,89,386).  visible(fcgt_2,386). happensAt(running(fcgt_2),386). orientation(fcgt_3,164,386). coords(fcgt_3,99,188,386).  visible(fcgt_3,386). happensAt(walking(fcgt_3),386).
orientation(fcgt_2,142,387). coords(fcgt_2,80,89,387).  visible(fcgt_2,387). happensAt(running(fcgt_2),387). orientation(fcgt_3,164,387). coords(fcgt_3,99,188,387).  visible(fcgt_3,387). happensAt(walking(fcgt_3),387).
orientation(fcgt_2,142,388). coords(fcgt_2,79,88,388).  visible(fcgt_2,388). happensAt(running(fcgt_2),388). orientation(fcgt_3,164,388). coords(fcgt_3,99,189,388).  visible(fcgt_3,388). happensAt(walking(fcgt_3),388).
orientation(fcgt_2,142,389). coords(fcgt_2,79,87,389).  visible(fcgt_2,389). happensAt(running(fcgt_2),389). orientation(fcgt_3,164,389). coords(fcgt_3,99,189,389).  visible(fcgt_3,389). happensAt(walking(fcgt_3),389).
orientation(fcgt_2,139,390). coords(fcgt_2,79,87,390).  visible(fcgt_2,390). happensAt(running(fcgt_2),390). orientation(fcgt_3,164,390). coords(fcgt_3,100,190,390).  visible(fcgt_3,390). happensAt(walking(fcgt_3),390).
#show holdsAt/2.