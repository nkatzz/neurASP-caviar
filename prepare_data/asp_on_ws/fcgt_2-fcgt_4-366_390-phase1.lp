
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
orientation(fcgt_2,142,379). coords(fcgt_2,84,93,379).  visible(fcgt_2,379). happensAt(running(fcgt_2),379). orientation(fcgt_4,166,379). coords(fcgt_4,82,206,379).  visible(fcgt_4,379). happensAt(inactive(fcgt_4),379).
orientation(fcgt_2,142,380). coords(fcgt_2,84,93,380).  visible(fcgt_2,380). happensAt(running(fcgt_2),380). orientation(fcgt_4,166,380). coords(fcgt_4,82,206,380).  visible(fcgt_4,380). happensAt(inactive(fcgt_4),380).
orientation(fcgt_2,142,381). coords(fcgt_2,84,93,381).  visible(fcgt_2,381). happensAt(running(fcgt_2),381). orientation(fcgt_4,166,381). coords(fcgt_4,82,206,381).  visible(fcgt_4,381). happensAt(inactive(fcgt_4),381).
orientation(fcgt_2,142,382). coords(fcgt_2,83,92,382).  visible(fcgt_2,382). happensAt(running(fcgt_2),382). orientation(fcgt_4,166,382). coords(fcgt_4,82,206,382).  visible(fcgt_4,382). happensAt(inactive(fcgt_4),382).
orientation(fcgt_2,142,383). coords(fcgt_2,83,92,383).  visible(fcgt_2,383). happensAt(running(fcgt_2),383). orientation(fcgt_4,166,383). coords(fcgt_4,82,206,383).  visible(fcgt_4,383). happensAt(inactive(fcgt_4),383).
orientation(fcgt_2,142,384). coords(fcgt_2,82,91,384).  visible(fcgt_2,384). happensAt(running(fcgt_2),384). orientation(fcgt_4,166,384). coords(fcgt_4,82,206,384).  visible(fcgt_4,384). happensAt(inactive(fcgt_4),384).
orientation(fcgt_2,142,385). coords(fcgt_2,81,90,385).  visible(fcgt_2,385). happensAt(running(fcgt_2),385). orientation(fcgt_4,166,385). coords(fcgt_4,82,206,385).  visible(fcgt_4,385). happensAt(inactive(fcgt_4),385).
orientation(fcgt_2,142,386). coords(fcgt_2,81,89,386).  visible(fcgt_2,386). happensAt(running(fcgt_2),386). orientation(fcgt_4,166,386). coords(fcgt_4,82,206,386).  visible(fcgt_4,386). happensAt(inactive(fcgt_4),386).
orientation(fcgt_2,142,387). coords(fcgt_2,80,89,387).  visible(fcgt_2,387). happensAt(running(fcgt_2),387). orientation(fcgt_4,166,387). coords(fcgt_4,82,206,387).  visible(fcgt_4,387). happensAt(inactive(fcgt_4),387).
orientation(fcgt_2,142,388). coords(fcgt_2,79,88,388).  visible(fcgt_2,388). happensAt(running(fcgt_2),388). orientation(fcgt_4,166,388). coords(fcgt_4,82,206,388).  visible(fcgt_4,388). happensAt(inactive(fcgt_4),388).
orientation(fcgt_2,142,389). coords(fcgt_2,79,87,389).  visible(fcgt_2,389). happensAt(running(fcgt_2),389). orientation(fcgt_4,166,389). coords(fcgt_4,82,206,389).  visible(fcgt_4,389). happensAt(inactive(fcgt_4),389).
orientation(fcgt_2,139,390). coords(fcgt_2,79,87,390).  visible(fcgt_2,390). happensAt(running(fcgt_2),390). orientation(fcgt_4,166,390). coords(fcgt_4,82,206,390).  visible(fcgt_4,390). happensAt(inactive(fcgt_4),390).
#show holdsAt/2.