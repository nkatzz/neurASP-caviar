
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

	orientation(fomdgt3_2,42,366). coords(fomdgt3_2,262,95,366).  visible(fomdgt3_2,366). happensAt(inactive(fomdgt3_2),366). orientation(fomdgt3_6,156,366). coords(fomdgt3_6,155,94,366).  visible(fomdgt3_6,366). happensAt(running(fomdgt3_6),366).
orientation(fomdgt3_2,42,367). coords(fomdgt3_2,262,95,367).  visible(fomdgt3_2,367). happensAt(inactive(fomdgt3_2),367). orientation(fomdgt3_6,156,367). coords(fomdgt3_6,152,94,367).  visible(fomdgt3_6,367). happensAt(running(fomdgt3_6),367).
orientation(fomdgt3_2,42,368). coords(fomdgt3_2,262,95,368).  visible(fomdgt3_2,368). happensAt(inactive(fomdgt3_2),368). orientation(fomdgt3_6,156,368). coords(fomdgt3_6,148,93,368).  visible(fomdgt3_6,368). happensAt(running(fomdgt3_6),368).
orientation(fomdgt3_2,42,369). coords(fomdgt3_2,262,95,369).  visible(fomdgt3_2,369). happensAt(inactive(fomdgt3_2),369). orientation(fomdgt3_6,156,369). coords(fomdgt3_6,147,93,369).  visible(fomdgt3_6,369). happensAt(running(fomdgt3_6),369).
orientation(fomdgt3_2,42,370). coords(fomdgt3_2,262,95,370).  visible(fomdgt3_2,370). happensAt(inactive(fomdgt3_2),370). orientation(fomdgt3_6,156,370). coords(fomdgt3_6,140,94,370).  visible(fomdgt3_6,370). happensAt(running(fomdgt3_6),370).
orientation(fomdgt3_2,42,371). coords(fomdgt3_2,262,97,371).  visible(fomdgt3_2,371). happensAt(inactive(fomdgt3_2),371). orientation(fomdgt3_6,156,371). coords(fomdgt3_6,136,94,371).  visible(fomdgt3_6,371). happensAt(running(fomdgt3_6),371).
orientation(fomdgt3_2,42,372). coords(fomdgt3_2,262,96,372).  visible(fomdgt3_2,372). happensAt(inactive(fomdgt3_2),372). orientation(fomdgt3_6,156,372). coords(fomdgt3_6,134,94,372).  visible(fomdgt3_6,372). happensAt(running(fomdgt3_6),372).
orientation(fomdgt3_2,42,373). coords(fomdgt3_2,261,96,373).  visible(fomdgt3_2,373). happensAt(inactive(fomdgt3_2),373). orientation(fomdgt3_6,156,373). coords(fomdgt3_6,132,93,373).  visible(fomdgt3_6,373). happensAt(running(fomdgt3_6),373).
orientation(fomdgt3_2,42,374). coords(fomdgt3_2,262,96,374).  visible(fomdgt3_2,374). happensAt(inactive(fomdgt3_2),374). orientation(fomdgt3_6,156,374). coords(fomdgt3_6,130,92,374).  visible(fomdgt3_6,374). happensAt(running(fomdgt3_6),374).
orientation(fomdgt3_2,42,375). coords(fomdgt3_2,262,96,375).  visible(fomdgt3_2,375). happensAt(inactive(fomdgt3_2),375). orientation(fomdgt3_6,156,375). coords(fomdgt3_6,127,91,375).  visible(fomdgt3_6,375). happensAt(running(fomdgt3_6),375).
orientation(fomdgt3_2,42,376). coords(fomdgt3_2,262,96,376).  visible(fomdgt3_2,376). happensAt(inactive(fomdgt3_2),376). orientation(fomdgt3_6,156,376). coords(fomdgt3_6,124,91,376).  visible(fomdgt3_6,376). happensAt(running(fomdgt3_6),376).
orientation(fomdgt3_2,42,377). coords(fomdgt3_2,262,96,377).  visible(fomdgt3_2,377). happensAt(inactive(fomdgt3_2),377). orientation(fomdgt3_6,156,377). coords(fomdgt3_6,120,91,377).  visible(fomdgt3_6,377). happensAt(running(fomdgt3_6),377).
orientation(fomdgt3_2,42,378). coords(fomdgt3_2,262,96,378).  visible(fomdgt3_2,378). happensAt(inactive(fomdgt3_2),378). orientation(fomdgt3_6,156,378). coords(fomdgt3_6,118,91,378).  visible(fomdgt3_6,378). happensAt(running(fomdgt3_6),378).
orientation(fomdgt3_2,42,379). coords(fomdgt3_2,262,96,379).  visible(fomdgt3_2,379). happensAt(inactive(fomdgt3_2),379). orientation(fomdgt3_6,156,379). coords(fomdgt3_6,116,91,379).  visible(fomdgt3_6,379). happensAt(running(fomdgt3_6),379).
orientation(fomdgt3_2,42,380). coords(fomdgt3_2,262,96,380).  visible(fomdgt3_2,380). happensAt(inactive(fomdgt3_2),380). orientation(fomdgt3_6,156,380). coords(fomdgt3_6,115,91,380).  visible(fomdgt3_6,380). happensAt(running(fomdgt3_6),380).
orientation(fomdgt3_2,42,381). coords(fomdgt3_2,262,96,381).  visible(fomdgt3_2,381). happensAt(inactive(fomdgt3_2),381). orientation(fomdgt3_6,156,381). coords(fomdgt3_6,114,90,381).  visible(fomdgt3_6,381). happensAt(running(fomdgt3_6),381).
orientation(fomdgt3_2,42,382). coords(fomdgt3_2,262,96,382).  visible(fomdgt3_2,382). happensAt(inactive(fomdgt3_2),382). orientation(fomdgt3_6,156,382). coords(fomdgt3_6,113,90,382).  visible(fomdgt3_6,382). happensAt(running(fomdgt3_6),382).
orientation(fomdgt3_2,42,383). coords(fomdgt3_2,262,96,383).  visible(fomdgt3_2,383). happensAt(inactive(fomdgt3_2),383). orientation(fomdgt3_6,156,383). coords(fomdgt3_6,108,89,383).  visible(fomdgt3_6,383). happensAt(running(fomdgt3_6),383).
orientation(fomdgt3_2,42,384). coords(fomdgt3_2,262,96,384).  visible(fomdgt3_2,384). happensAt(inactive(fomdgt3_2),384). orientation(fomdgt3_6,140,384). coords(fomdgt3_6,106,89,384).  visible(fomdgt3_6,384). happensAt(running(fomdgt3_6),384).
orientation(fomdgt3_2,42,385). coords(fomdgt3_2,262,96,385).  visible(fomdgt3_2,385). happensAt(inactive(fomdgt3_2),385). orientation(fomdgt3_6,140,385). coords(fomdgt3_6,103,88,385).  visible(fomdgt3_6,385). happensAt(running(fomdgt3_6),385).
orientation(fomdgt3_2,42,386). coords(fomdgt3_2,262,96,386).  visible(fomdgt3_2,386). happensAt(inactive(fomdgt3_2),386). orientation(fomdgt3_6,140,386). coords(fomdgt3_6,102,88,386).  visible(fomdgt3_6,386). happensAt(running(fomdgt3_6),386).
orientation(fomdgt3_2,42,387). coords(fomdgt3_2,262,96,387).  visible(fomdgt3_2,387). happensAt(inactive(fomdgt3_2),387). orientation(fomdgt3_6,140,387). coords(fomdgt3_6,100,88,387).  visible(fomdgt3_6,387). happensAt(running(fomdgt3_6),387).
orientation(fomdgt3_2,42,388). coords(fomdgt3_2,262,96,388).  visible(fomdgt3_2,388). happensAt(inactive(fomdgt3_2),388). orientation(fomdgt3_6,140,388). coords(fomdgt3_6,100,88,388).  visible(fomdgt3_6,388). happensAt(running(fomdgt3_6),388).
orientation(fomdgt3_2,42,389). coords(fomdgt3_2,262,96,389).  visible(fomdgt3_2,389). happensAt(inactive(fomdgt3_2),389). orientation(fomdgt3_6,140,389). coords(fomdgt3_6,99,88,389).  visible(fomdgt3_6,389). happensAt(running(fomdgt3_6),389).
orientation(fomdgt3_2,42,390). coords(fomdgt3_2,262,96,390).  visible(fomdgt3_2,390). happensAt(inactive(fomdgt3_2),390). orientation(fomdgt3_6,140,390). coords(fomdgt3_6,96,85,390).  visible(fomdgt3_6,390). happensAt(running(fomdgt3_6),390).
#show holdsAt/2.