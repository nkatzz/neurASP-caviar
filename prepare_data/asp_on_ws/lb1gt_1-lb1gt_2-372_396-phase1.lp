
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

	orientation(lb1gt_1,151,372). coords(lb1gt_1,167,170,372).  visible(lb1gt_1,372). happensAt(active(lb1gt_1),372). orientation(lb1gt_2,146,372). coords(lb1gt_2,191,142,372).  visible(lb1gt_2,372). happensAt(active(lb1gt_2),372).
orientation(lb1gt_1,151,373). coords(lb1gt_1,167,170,373).  visible(lb1gt_1,373). happensAt(active(lb1gt_1),373). orientation(lb1gt_2,144,373). coords(lb1gt_2,191,142,373).  visible(lb1gt_2,373). happensAt(active(lb1gt_2),373).
orientation(lb1gt_1,151,374). coords(lb1gt_1,167,170,374).  visible(lb1gt_1,374). happensAt(active(lb1gt_1),374). orientation(lb1gt_2,144,374). coords(lb1gt_2,191,141,374).  visible(lb1gt_2,374). happensAt(active(lb1gt_2),374).
orientation(lb1gt_1,151,375). coords(lb1gt_1,167,170,375).  visible(lb1gt_1,375). happensAt(active(lb1gt_1),375). orientation(lb1gt_2,144,375). coords(lb1gt_2,191,141,375).  visible(lb1gt_2,375). happensAt(active(lb1gt_2),375).
orientation(lb1gt_1,151,376). coords(lb1gt_1,167,170,376).  visible(lb1gt_1,376). happensAt(active(lb1gt_1),376). orientation(lb1gt_2,144,376). coords(lb1gt_2,191,141,376).  visible(lb1gt_2,376). happensAt(active(lb1gt_2),376).
orientation(lb1gt_1,151,377). coords(lb1gt_1,167,170,377).  visible(lb1gt_1,377). happensAt(active(lb1gt_1),377). orientation(lb1gt_2,144,377). coords(lb1gt_2,191,141,377).  visible(lb1gt_2,377). happensAt(active(lb1gt_2),377).
orientation(lb1gt_1,151,378). coords(lb1gt_1,167,169,378).  visible(lb1gt_1,378). happensAt(active(lb1gt_1),378). orientation(lb1gt_2,144,378). coords(lb1gt_2,191,141,378).  visible(lb1gt_2,378). happensAt(active(lb1gt_2),378).
orientation(lb1gt_1,151,379). coords(lb1gt_1,167,169,379).  visible(lb1gt_1,379). happensAt(active(lb1gt_1),379). orientation(lb1gt_2,144,379). coords(lb1gt_2,191,140,379).  visible(lb1gt_2,379). happensAt(active(lb1gt_2),379).
orientation(lb1gt_1,151,380). coords(lb1gt_1,167,169,380).  visible(lb1gt_1,380). happensAt(active(lb1gt_1),380). orientation(lb1gt_2,144,380). coords(lb1gt_2,191,140,380).  visible(lb1gt_2,380). happensAt(active(lb1gt_2),380).
orientation(lb1gt_1,151,381). coords(lb1gt_1,167,169,381).  visible(lb1gt_1,381). happensAt(active(lb1gt_1),381). orientation(lb1gt_2,144,381). coords(lb1gt_2,191,140,381).  visible(lb1gt_2,381). happensAt(active(lb1gt_2),381).
orientation(lb1gt_1,151,382). coords(lb1gt_1,166,169,382).  visible(lb1gt_1,382). happensAt(active(lb1gt_1),382). orientation(lb1gt_2,144,382). coords(lb1gt_2,190,140,382).  visible(lb1gt_2,382). happensAt(active(lb1gt_2),382).
orientation(lb1gt_1,151,383). coords(lb1gt_1,166,169,383).  visible(lb1gt_1,383). happensAt(active(lb1gt_1),383). orientation(lb1gt_2,144,383). coords(lb1gt_2,190,140,383).  visible(lb1gt_2,383). happensAt(active(lb1gt_2),383).
orientation(lb1gt_1,151,384). coords(lb1gt_1,166,168,384).  visible(lb1gt_1,384). happensAt(active(lb1gt_1),384). orientation(lb1gt_2,144,384). coords(lb1gt_2,190,141,384).  visible(lb1gt_2,384). happensAt(active(lb1gt_2),384).
orientation(lb1gt_1,151,385). coords(lb1gt_1,166,168,385).  visible(lb1gt_1,385). happensAt(active(lb1gt_1),385). orientation(lb1gt_2,144,385). coords(lb1gt_2,190,141,385).  visible(lb1gt_2,385). happensAt(active(lb1gt_2),385).
orientation(lb1gt_1,151,386). coords(lb1gt_1,166,168,386).  visible(lb1gt_1,386). happensAt(active(lb1gt_1),386). orientation(lb1gt_2,144,386). coords(lb1gt_2,190,141,386).  visible(lb1gt_2,386). happensAt(active(lb1gt_2),386).
orientation(lb1gt_1,151,387). coords(lb1gt_1,166,168,387).  visible(lb1gt_1,387). happensAt(active(lb1gt_1),387). orientation(lb1gt_2,144,387). coords(lb1gt_2,190,141,387).  visible(lb1gt_2,387). happensAt(active(lb1gt_2),387).
orientation(lb1gt_1,151,388). coords(lb1gt_1,166,168,388).  visible(lb1gt_1,388). happensAt(active(lb1gt_1),388). orientation(lb1gt_2,144,388). coords(lb1gt_2,190,141,388).  visible(lb1gt_2,388). happensAt(active(lb1gt_2),388).
orientation(lb1gt_1,151,389). coords(lb1gt_1,166,168,389).  visible(lb1gt_1,389). happensAt(active(lb1gt_1),389). orientation(lb1gt_2,144,389). coords(lb1gt_2,190,141,389).  visible(lb1gt_2,389). happensAt(active(lb1gt_2),389).
orientation(lb1gt_1,151,390). coords(lb1gt_1,166,169,390).  visible(lb1gt_1,390). happensAt(active(lb1gt_1),390). orientation(lb1gt_2,144,390). coords(lb1gt_2,190,141,390).  visible(lb1gt_2,390). happensAt(active(lb1gt_2),390).
orientation(lb1gt_1,151,391). coords(lb1gt_1,166,170,391).  visible(lb1gt_1,391). happensAt(active(lb1gt_1),391). orientation(lb1gt_2,144,391). coords(lb1gt_2,190,141,391).  visible(lb1gt_2,391). happensAt(active(lb1gt_2),391).
orientation(lb1gt_1,151,392). coords(lb1gt_1,166,171,392).  visible(lb1gt_1,392). happensAt(active(lb1gt_1),392). orientation(lb1gt_2,144,392). coords(lb1gt_2,190,141,392).  visible(lb1gt_2,392). happensAt(active(lb1gt_2),392).
orientation(lb1gt_1,151,393). coords(lb1gt_1,166,171,393).  visible(lb1gt_1,393). happensAt(active(lb1gt_1),393). orientation(lb1gt_2,144,393). coords(lb1gt_2,190,141,393).  visible(lb1gt_2,393). happensAt(active(lb1gt_2),393).
orientation(lb1gt_1,151,394). coords(lb1gt_1,166,171,394).  visible(lb1gt_1,394). happensAt(active(lb1gt_1),394). orientation(lb1gt_2,144,394). coords(lb1gt_2,190,141,394).  visible(lb1gt_2,394). happensAt(active(lb1gt_2),394).
orientation(lb1gt_1,151,395). coords(lb1gt_1,166,171,395).  visible(lb1gt_1,395). happensAt(active(lb1gt_1),395). orientation(lb1gt_2,144,395). coords(lb1gt_2,190,141,395).  visible(lb1gt_2,395). happensAt(active(lb1gt_2),395).
orientation(lb1gt_1,151,396). coords(lb1gt_1,166,170,396).  visible(lb1gt_1,396). happensAt(active(lb1gt_1),396). orientation(lb1gt_2,144,396). coords(lb1gt_2,189,141,396).  visible(lb1gt_2,396). happensAt(active(lb1gt_2),396).
#show holdsAt/2.