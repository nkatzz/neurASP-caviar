
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

	orientation(mwt2gt_1,151,373). coords(mwt2gt_1,125,83,373).  visible(mwt2gt_1,373). happensAt(walking(mwt2gt_1),373). orientation(mwt2gt_2,151,373). coords(mwt2gt_2,128,96,373).  visible(mwt2gt_2,373). happensAt(walking(mwt2gt_2),373).
orientation(mwt2gt_1,153,374). coords(mwt2gt_1,124,83,374).  visible(mwt2gt_1,374). happensAt(walking(mwt2gt_1),374). orientation(mwt2gt_2,151,374). coords(mwt2gt_2,128,96,374).  visible(mwt2gt_2,374). happensAt(walking(mwt2gt_2),374).
orientation(mwt2gt_1,153,375). coords(mwt2gt_1,123,84,375).  visible(mwt2gt_1,375). happensAt(walking(mwt2gt_1),375). orientation(mwt2gt_2,151,375). coords(mwt2gt_2,127,97,375).  visible(mwt2gt_2,375). happensAt(walking(mwt2gt_2),375).
orientation(mwt2gt_1,153,376). coords(mwt2gt_1,122,84,376).  visible(mwt2gt_1,376). happensAt(walking(mwt2gt_1),376). orientation(mwt2gt_2,151,376). coords(mwt2gt_2,127,97,376).  visible(mwt2gt_2,376). happensAt(walking(mwt2gt_2),376).
orientation(mwt2gt_1,153,377). coords(mwt2gt_1,121,84,377).  visible(mwt2gt_1,377). happensAt(walking(mwt2gt_1),377). orientation(mwt2gt_2,151,377). coords(mwt2gt_2,127,98,377).  visible(mwt2gt_2,377). happensAt(walking(mwt2gt_2),377).
orientation(mwt2gt_1,153,378). coords(mwt2gt_1,119,84,378).  visible(mwt2gt_1,378). happensAt(walking(mwt2gt_1),378). orientation(mwt2gt_2,151,378). coords(mwt2gt_2,126,99,378).  visible(mwt2gt_2,378). happensAt(walking(mwt2gt_2),378).
orientation(mwt2gt_1,153,379). coords(mwt2gt_1,119,84,379).  visible(mwt2gt_1,379). happensAt(walking(mwt2gt_1),379). orientation(mwt2gt_2,151,379). coords(mwt2gt_2,126,99,379).  visible(mwt2gt_2,379). happensAt(walking(mwt2gt_2),379).
orientation(mwt2gt_1,153,380). coords(mwt2gt_1,119,85,380).  visible(mwt2gt_1,380). happensAt(walking(mwt2gt_1),380). orientation(mwt2gt_2,151,380). coords(mwt2gt_2,124,101,380).  visible(mwt2gt_2,380). happensAt(walking(mwt2gt_2),380).
orientation(mwt2gt_1,153,381). coords(mwt2gt_1,119,85,381).  visible(mwt2gt_1,381). happensAt(walking(mwt2gt_1),381). orientation(mwt2gt_2,151,381). coords(mwt2gt_2,123,101,381).  visible(mwt2gt_2,381). happensAt(walking(mwt2gt_2),381).
orientation(mwt2gt_1,153,382). coords(mwt2gt_1,118,85,382).  visible(mwt2gt_1,382). happensAt(walking(mwt2gt_1),382). orientation(mwt2gt_2,151,382). coords(mwt2gt_2,121,101,382).  visible(mwt2gt_2,382). happensAt(walking(mwt2gt_2),382).
orientation(mwt2gt_1,153,383). coords(mwt2gt_1,118,85,383).  visible(mwt2gt_1,383). happensAt(walking(mwt2gt_1),383). orientation(mwt2gt_2,151,383). coords(mwt2gt_2,118,101,383).  visible(mwt2gt_2,383). happensAt(walking(mwt2gt_2),383).
orientation(mwt2gt_1,153,384). coords(mwt2gt_1,118,85,384).  visible(mwt2gt_1,384). happensAt(walking(mwt2gt_1),384). orientation(mwt2gt_2,151,384). coords(mwt2gt_2,117,102,384).  visible(mwt2gt_2,384). happensAt(walking(mwt2gt_2),384).
orientation(mwt2gt_1,153,385). coords(mwt2gt_1,118,85,385).  visible(mwt2gt_1,385). happensAt(walking(mwt2gt_1),385). orientation(mwt2gt_2,151,385). coords(mwt2gt_2,115,104,385).  visible(mwt2gt_2,385). happensAt(walking(mwt2gt_2),385).
orientation(mwt2gt_1,153,386). coords(mwt2gt_1,117,85,386).  visible(mwt2gt_1,386). happensAt(walking(mwt2gt_1),386). orientation(mwt2gt_2,151,386). coords(mwt2gt_2,115,105,386).  visible(mwt2gt_2,386). happensAt(walking(mwt2gt_2),386).
orientation(mwt2gt_1,153,387). coords(mwt2gt_1,116,85,387).  visible(mwt2gt_1,387). happensAt(walking(mwt2gt_1),387). orientation(mwt2gt_2,151,387). coords(mwt2gt_2,115,107,387).  visible(mwt2gt_2,387). happensAt(walking(mwt2gt_2),387).
orientation(mwt2gt_1,153,388). coords(mwt2gt_1,115,85,388).  visible(mwt2gt_1,388). happensAt(walking(mwt2gt_1),388). orientation(mwt2gt_2,144,388). coords(mwt2gt_2,114,108,388).  visible(mwt2gt_2,388). happensAt(walking(mwt2gt_2),388).
orientation(mwt2gt_1,153,389). coords(mwt2gt_1,113,86,389).  visible(mwt2gt_1,389). happensAt(walking(mwt2gt_1),389). orientation(mwt2gt_2,144,389). coords(mwt2gt_2,114,109,389).  visible(mwt2gt_2,389). happensAt(walking(mwt2gt_2),389).
orientation(mwt2gt_1,149,390). coords(mwt2gt_1,112,86,390).  visible(mwt2gt_1,390). happensAt(walking(mwt2gt_1),390). orientation(mwt2gt_2,144,390). coords(mwt2gt_2,113,110,390).  visible(mwt2gt_2,390). happensAt(walking(mwt2gt_2),390).
orientation(mwt2gt_1,149,391). coords(mwt2gt_1,111,87,391).  visible(mwt2gt_1,391). happensAt(walking(mwt2gt_1),391). orientation(mwt2gt_2,144,391). coords(mwt2gt_2,113,110,391).  visible(mwt2gt_2,391). happensAt(walking(mwt2gt_2),391).
orientation(mwt2gt_1,149,392). coords(mwt2gt_1,111,88,392).  visible(mwt2gt_1,392). happensAt(walking(mwt2gt_1),392). orientation(mwt2gt_2,144,392). coords(mwt2gt_2,112,110,392).  visible(mwt2gt_2,392). happensAt(walking(mwt2gt_2),392).
orientation(mwt2gt_1,149,393). coords(mwt2gt_1,110,88,393).  visible(mwt2gt_1,393). happensAt(walking(mwt2gt_1),393). orientation(mwt2gt_2,144,393). coords(mwt2gt_2,111,111,393).  visible(mwt2gt_2,393). happensAt(walking(mwt2gt_2),393).
orientation(mwt2gt_1,146,394). coords(mwt2gt_1,111,88,394).  visible(mwt2gt_1,394). happensAt(walking(mwt2gt_1),394). orientation(mwt2gt_2,144,394). coords(mwt2gt_2,110,112,394).  visible(mwt2gt_2,394). happensAt(walking(mwt2gt_2),394).
orientation(mwt2gt_1,146,395). coords(mwt2gt_1,110,89,395).  visible(mwt2gt_1,395). happensAt(walking(mwt2gt_1),395). orientation(mwt2gt_2,144,395). coords(mwt2gt_2,108,112,395).  visible(mwt2gt_2,395). happensAt(walking(mwt2gt_2),395).
orientation(mwt2gt_1,146,396). coords(mwt2gt_1,110,89,396).  visible(mwt2gt_1,396). happensAt(walking(mwt2gt_1),396). orientation(mwt2gt_2,144,396). coords(mwt2gt_2,106,112,396).  visible(mwt2gt_2,396). happensAt(walking(mwt2gt_2),396).
orientation(mwt2gt_1,146,397). coords(mwt2gt_1,109,89,397).  visible(mwt2gt_1,397). happensAt(walking(mwt2gt_1),397). orientation(mwt2gt_2,144,397). coords(mwt2gt_2,104,113,397).  visible(mwt2gt_2,397). happensAt(walking(mwt2gt_2),397).
#show holdsAt/2.