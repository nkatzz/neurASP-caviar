
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

	orientation(lbgt_0,1,369). coords(lbgt_0,239,195,369).  visible(lbgt_0,369). happensAt(walking(lbgt_0),369). orientation(lbgt_4,147,369). coords(lbgt_4,81,63,369).  happensAt(appear(lbgt_4),369). happensAt(walking(lbgt_4),369).
orientation(lbgt_0,1,370). coords(lbgt_0,241,197,370).  visible(lbgt_0,370). happensAt(walking(lbgt_0),370). orientation(lbgt_4,147,370). coords(lbgt_4,81,63,370).  visible(lbgt_4,370). happensAt(walking(lbgt_4),370).
orientation(lbgt_0,1,371). coords(lbgt_0,243,200,371).  visible(lbgt_0,371). happensAt(walking(lbgt_0),371). orientation(lbgt_4,147,371). coords(lbgt_4,81,63,371).  visible(lbgt_4,371). happensAt(walking(lbgt_4),371).
orientation(lbgt_0,2,372). coords(lbgt_0,245,203,372).  visible(lbgt_0,372). happensAt(walking(lbgt_0),372). orientation(lbgt_4,147,372). coords(lbgt_4,81,63,372).  visible(lbgt_4,372). happensAt(walking(lbgt_4),372).
orientation(lbgt_0,2,373). coords(lbgt_0,245,206,373).  visible(lbgt_0,373). happensAt(walking(lbgt_0),373). orientation(lbgt_4,147,373). coords(lbgt_4,81,63,373).  visible(lbgt_4,373). happensAt(walking(lbgt_4),373).
orientation(lbgt_0,2,374). coords(lbgt_0,247,211,374).  visible(lbgt_0,374). happensAt(walking(lbgt_0),374). orientation(lbgt_4,147,374). coords(lbgt_4,81,63,374).  visible(lbgt_4,374). happensAt(walking(lbgt_4),374).
orientation(lbgt_0,2,375). coords(lbgt_0,247,215,375).  visible(lbgt_0,375). happensAt(walking(lbgt_0),375). orientation(lbgt_4,147,375). coords(lbgt_4,82,64,375).  visible(lbgt_4,375). happensAt(walking(lbgt_4),375).
orientation(lbgt_0,2,376). coords(lbgt_0,248,218,376).  visible(lbgt_0,376). happensAt(walking(lbgt_0),376). orientation(lbgt_4,147,376). coords(lbgt_4,82,65,376).  visible(lbgt_4,376). happensAt(walking(lbgt_4),376).
orientation(lbgt_0,4,377). coords(lbgt_0,251,222,377).  visible(lbgt_0,377). happensAt(walking(lbgt_0),377). orientation(lbgt_4,147,377). coords(lbgt_4,82,66,377).  visible(lbgt_4,377). happensAt(walking(lbgt_4),377).
orientation(lbgt_0,11,378). coords(lbgt_0,253,225,378).  visible(lbgt_0,378). happensAt(walking(lbgt_0),378). orientation(lbgt_4,147,378). coords(lbgt_4,82,66,378).  visible(lbgt_4,378). happensAt(walking(lbgt_4),378).
orientation(lbgt_0,11,379). coords(lbgt_0,256,227,379).  visible(lbgt_0,379). happensAt(walking(lbgt_0),379). orientation(lbgt_4,147,379). coords(lbgt_4,81,66,379).  visible(lbgt_4,379). happensAt(walking(lbgt_4),379).
orientation(lbgt_0,30,380). coords(lbgt_0,257,228,380).  visible(lbgt_0,380). happensAt(walking(lbgt_0),380). orientation(lbgt_4,147,380). coords(lbgt_4,81,67,380).  visible(lbgt_4,380). happensAt(walking(lbgt_4),380).
orientation(lbgt_0,30,381). coords(lbgt_0,259,229,381).  visible(lbgt_0,381). happensAt(walking(lbgt_0),381). orientation(lbgt_4,147,381). coords(lbgt_4,81,67,381).  visible(lbgt_4,381). happensAt(walking(lbgt_4),381).
orientation(lbgt_0,30,382). coords(lbgt_0,260,231,382).  visible(lbgt_0,382). happensAt(walking(lbgt_0),382). orientation(lbgt_4,147,382). coords(lbgt_4,81,67,382).  visible(lbgt_4,382). happensAt(walking(lbgt_4),382).
orientation(lbgt_0,30,383). coords(lbgt_0,261,233,383).  visible(lbgt_0,383). happensAt(walking(lbgt_0),383). orientation(lbgt_4,147,383). coords(lbgt_4,81,67,383).  visible(lbgt_4,383). happensAt(walking(lbgt_4),383).
orientation(lbgt_0,30,384). coords(lbgt_0,262,237,384).  visible(lbgt_0,384). happensAt(walking(lbgt_0),384). orientation(lbgt_4,147,384). coords(lbgt_4,80,67,384).  visible(lbgt_4,384). happensAt(walking(lbgt_4),384).
orientation(lbgt_0,34,385). coords(lbgt_0,264,242,385).  visible(lbgt_0,385). happensAt(walking(lbgt_0),385). orientation(lbgt_4,147,385). coords(lbgt_4,80,67,385).  visible(lbgt_4,385). happensAt(walking(lbgt_4),385).
orientation(lbgt_0,34,386). coords(lbgt_0,266,246,386).  visible(lbgt_0,386). happensAt(walking(lbgt_0),386). orientation(lbgt_4,147,386). coords(lbgt_4,80,67,386).  visible(lbgt_4,386). happensAt(walking(lbgt_4),386).
orientation(lbgt_0,34,387). coords(lbgt_0,267,251,387).  visible(lbgt_0,387). happensAt(walking(lbgt_0),387). orientation(lbgt_4,147,387). coords(lbgt_4,80,68,387).  visible(lbgt_4,387). happensAt(walking(lbgt_4),387).
orientation(lbgt_0,34,388). coords(lbgt_0,269,252,388).  visible(lbgt_0,388). happensAt(walking(lbgt_0),388). orientation(lbgt_4,147,388). coords(lbgt_4,80,68,388).  visible(lbgt_4,388). happensAt(walking(lbgt_4),388).
orientation(lbgt_0,34,389). coords(lbgt_0,270,254,389).  visible(lbgt_0,389). happensAt(walking(lbgt_0),389). orientation(lbgt_4,147,389). coords(lbgt_4,80,69,389).  visible(lbgt_4,389). happensAt(walking(lbgt_4),389).
orientation(lbgt_0,34,390). coords(lbgt_0,271,256,390).  visible(lbgt_0,390). happensAt(walking(lbgt_0),390). orientation(lbgt_4,147,390). coords(lbgt_4,80,70,390).  visible(lbgt_4,390). happensAt(walking(lbgt_4),390).
orientation(lbgt_0,39,391). coords(lbgt_0,272,257,391).  visible(lbgt_0,391). happensAt(walking(lbgt_0),391). orientation(lbgt_4,147,391). coords(lbgt_4,80,71,391).  visible(lbgt_4,391). happensAt(walking(lbgt_4),391).
orientation(lbgt_0,47,392). coords(lbgt_0,273,258,392).  visible(lbgt_0,392). happensAt(walking(lbgt_0),392). orientation(lbgt_4,147,392). coords(lbgt_4,80,72,392).  visible(lbgt_4,392). happensAt(walking(lbgt_4),392).
orientation(lbgt_0,53,393). coords(lbgt_0,273,260,393).  visible(lbgt_0,393). happensAt(walking(lbgt_0),393). orientation(lbgt_4,147,393). coords(lbgt_4,80,72,393).  visible(lbgt_4,393). happensAt(walking(lbgt_4),393).
#show holdsAt/2.