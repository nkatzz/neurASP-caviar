
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

	orientation(fomdgt3_1,150,372). coords(fomdgt3_1,46,102,372).  visible(fomdgt3_1,372). happensAt(inactive(fomdgt3_1),372). orientation(fomdgt3_2,42,372). coords(fomdgt3_2,262,96,372).  visible(fomdgt3_2,372). happensAt(inactive(fomdgt3_2),372).
orientation(fomdgt3_1,150,373). coords(fomdgt3_1,46,102,373).  visible(fomdgt3_1,373). happensAt(inactive(fomdgt3_1),373). orientation(fomdgt3_2,42,373). coords(fomdgt3_2,261,96,373).  visible(fomdgt3_2,373). happensAt(inactive(fomdgt3_2),373).
orientation(fomdgt3_1,150,374). coords(fomdgt3_1,46,102,374).  visible(fomdgt3_1,374). happensAt(inactive(fomdgt3_1),374). orientation(fomdgt3_2,42,374). coords(fomdgt3_2,262,96,374).  visible(fomdgt3_2,374). happensAt(inactive(fomdgt3_2),374).
orientation(fomdgt3_1,150,375). coords(fomdgt3_1,46,102,375).  visible(fomdgt3_1,375). happensAt(inactive(fomdgt3_1),375). orientation(fomdgt3_2,42,375). coords(fomdgt3_2,262,96,375).  visible(fomdgt3_2,375). happensAt(inactive(fomdgt3_2),375).
orientation(fomdgt3_1,150,376). coords(fomdgt3_1,46,102,376).  visible(fomdgt3_1,376). happensAt(inactive(fomdgt3_1),376). orientation(fomdgt3_2,42,376). coords(fomdgt3_2,262,96,376).  visible(fomdgt3_2,376). happensAt(inactive(fomdgt3_2),376).
orientation(fomdgt3_1,150,377). coords(fomdgt3_1,46,102,377).  visible(fomdgt3_1,377). happensAt(inactive(fomdgt3_1),377). orientation(fomdgt3_2,42,377). coords(fomdgt3_2,262,96,377).  visible(fomdgt3_2,377). happensAt(inactive(fomdgt3_2),377).
orientation(fomdgt3_1,150,378). coords(fomdgt3_1,46,102,378).  visible(fomdgt3_1,378). happensAt(inactive(fomdgt3_1),378). orientation(fomdgt3_2,42,378). coords(fomdgt3_2,262,96,378).  visible(fomdgt3_2,378). happensAt(inactive(fomdgt3_2),378).
orientation(fomdgt3_1,150,379). coords(fomdgt3_1,46,102,379).  visible(fomdgt3_1,379). happensAt(inactive(fomdgt3_1),379). orientation(fomdgt3_2,42,379). coords(fomdgt3_2,262,96,379).  visible(fomdgt3_2,379). happensAt(inactive(fomdgt3_2),379).
orientation(fomdgt3_1,150,380). coords(fomdgt3_1,46,102,380).  visible(fomdgt3_1,380). happensAt(inactive(fomdgt3_1),380). orientation(fomdgt3_2,42,380). coords(fomdgt3_2,262,96,380).  visible(fomdgt3_2,380). happensAt(inactive(fomdgt3_2),380).
orientation(fomdgt3_1,150,381). coords(fomdgt3_1,46,102,381).  visible(fomdgt3_1,381). happensAt(inactive(fomdgt3_1),381). orientation(fomdgt3_2,42,381). coords(fomdgt3_2,262,96,381).  visible(fomdgt3_2,381). happensAt(inactive(fomdgt3_2),381).
orientation(fomdgt3_1,150,382). coords(fomdgt3_1,46,102,382).  visible(fomdgt3_1,382). happensAt(inactive(fomdgt3_1),382). orientation(fomdgt3_2,42,382). coords(fomdgt3_2,262,96,382).  visible(fomdgt3_2,382). happensAt(inactive(fomdgt3_2),382).
orientation(fomdgt3_1,150,383). coords(fomdgt3_1,46,102,383).  visible(fomdgt3_1,383). happensAt(inactive(fomdgt3_1),383). orientation(fomdgt3_2,42,383). coords(fomdgt3_2,262,96,383).  visible(fomdgt3_2,383). happensAt(inactive(fomdgt3_2),383).
orientation(fomdgt3_1,150,384). coords(fomdgt3_1,46,102,384).  visible(fomdgt3_1,384). happensAt(inactive(fomdgt3_1),384). orientation(fomdgt3_2,42,384). coords(fomdgt3_2,262,96,384).  visible(fomdgt3_2,384). happensAt(inactive(fomdgt3_2),384).
orientation(fomdgt3_1,150,385). coords(fomdgt3_1,46,102,385).  visible(fomdgt3_1,385). happensAt(inactive(fomdgt3_1),385). orientation(fomdgt3_2,42,385). coords(fomdgt3_2,262,96,385).  visible(fomdgt3_2,385). happensAt(inactive(fomdgt3_2),385).
orientation(fomdgt3_1,150,386). coords(fomdgt3_1,46,102,386).  visible(fomdgt3_1,386). happensAt(inactive(fomdgt3_1),386). orientation(fomdgt3_2,42,386). coords(fomdgt3_2,262,96,386).  visible(fomdgt3_2,386). happensAt(inactive(fomdgt3_2),386).
orientation(fomdgt3_1,150,387). coords(fomdgt3_1,46,102,387).  visible(fomdgt3_1,387). happensAt(inactive(fomdgt3_1),387). orientation(fomdgt3_2,42,387). coords(fomdgt3_2,262,96,387).  visible(fomdgt3_2,387). happensAt(inactive(fomdgt3_2),387).
orientation(fomdgt3_1,150,388). coords(fomdgt3_1,46,102,388).  visible(fomdgt3_1,388). happensAt(inactive(fomdgt3_1),388). orientation(fomdgt3_2,42,388). coords(fomdgt3_2,262,96,388).  visible(fomdgt3_2,388). happensAt(inactive(fomdgt3_2),388).
orientation(fomdgt3_1,150,389). coords(fomdgt3_1,46,102,389).  visible(fomdgt3_1,389). happensAt(inactive(fomdgt3_1),389). orientation(fomdgt3_2,42,389). coords(fomdgt3_2,262,96,389).  visible(fomdgt3_2,389). happensAt(inactive(fomdgt3_2),389).
orientation(fomdgt3_1,150,390). coords(fomdgt3_1,46,102,390).  visible(fomdgt3_1,390). happensAt(inactive(fomdgt3_1),390). orientation(fomdgt3_2,42,390). coords(fomdgt3_2,262,96,390).  visible(fomdgt3_2,390). happensAt(inactive(fomdgt3_2),390).
orientation(fomdgt3_1,150,391). coords(fomdgt3_1,46,102,391).  visible(fomdgt3_1,391). happensAt(inactive(fomdgt3_1),391). orientation(fomdgt3_2,42,391). coords(fomdgt3_2,262,96,391).  visible(fomdgt3_2,391). happensAt(inactive(fomdgt3_2),391).
orientation(fomdgt3_1,150,392). coords(fomdgt3_1,46,102,392).  visible(fomdgt3_1,392). happensAt(inactive(fomdgt3_1),392). orientation(fomdgt3_2,42,392). coords(fomdgt3_2,262,96,392).  visible(fomdgt3_2,392). happensAt(inactive(fomdgt3_2),392).
orientation(fomdgt3_1,150,393). coords(fomdgt3_1,46,102,393).  visible(fomdgt3_1,393). happensAt(inactive(fomdgt3_1),393). orientation(fomdgt3_2,42,393). coords(fomdgt3_2,262,96,393).  visible(fomdgt3_2,393). happensAt(inactive(fomdgt3_2),393).
orientation(fomdgt3_1,150,394). coords(fomdgt3_1,46,102,394).  visible(fomdgt3_1,394). happensAt(inactive(fomdgt3_1),394). orientation(fomdgt3_2,42,394). coords(fomdgt3_2,262,96,394).  visible(fomdgt3_2,394). happensAt(inactive(fomdgt3_2),394).
orientation(fomdgt3_1,150,395). coords(fomdgt3_1,46,102,395).  visible(fomdgt3_1,395). happensAt(inactive(fomdgt3_1),395). orientation(fomdgt3_2,42,395). coords(fomdgt3_2,262,96,395).  visible(fomdgt3_2,395). happensAt(inactive(fomdgt3_2),395).
orientation(fomdgt3_1,150,396). coords(fomdgt3_1,46,102,396).  visible(fomdgt3_1,396). happensAt(inactive(fomdgt3_1),396). orientation(fomdgt3_2,42,396). coords(fomdgt3_2,262,96,396).  visible(fomdgt3_2,396). happensAt(inactive(fomdgt3_2),396).
#show holdsAt/2.