
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

	orientation(ms3ggt_0,158,374). coords(ms3ggt_0,137,142,374).  visible(ms3ggt_0,374). happensAt(walking(ms3ggt_0),374). orientation(ms3ggt_1,156,374). coords(ms3ggt_1,130,158,374).  visible(ms3ggt_1,374). happensAt(walking(ms3ggt_1),374).
orientation(ms3ggt_0,158,375). coords(ms3ggt_0,137,142,375).  visible(ms3ggt_0,375). happensAt(walking(ms3ggt_0),375). orientation(ms3ggt_1,156,375). coords(ms3ggt_1,130,158,375).  visible(ms3ggt_1,375). happensAt(walking(ms3ggt_1),375).
orientation(ms3ggt_0,158,376). coords(ms3ggt_0,137,142,376).  visible(ms3ggt_0,376). happensAt(walking(ms3ggt_0),376). orientation(ms3ggt_1,156,376). coords(ms3ggt_1,130,158,376).  visible(ms3ggt_1,376). happensAt(walking(ms3ggt_1),376).
orientation(ms3ggt_0,158,377). coords(ms3ggt_0,139,142,377).  visible(ms3ggt_0,377). happensAt(active(ms3ggt_0),377). orientation(ms3ggt_1,156,377). coords(ms3ggt_1,132,158,377).  visible(ms3ggt_1,377). happensAt(walking(ms3ggt_1),377).
orientation(ms3ggt_0,158,378). coords(ms3ggt_0,140,142,378).  visible(ms3ggt_0,378). happensAt(active(ms3ggt_0),378). orientation(ms3ggt_1,156,378). coords(ms3ggt_1,134,158,378).  visible(ms3ggt_1,378). happensAt(walking(ms3ggt_1),378).
orientation(ms3ggt_0,158,379). coords(ms3ggt_0,140,142,379).  visible(ms3ggt_0,379). happensAt(active(ms3ggt_0),379). orientation(ms3ggt_1,162,379). coords(ms3ggt_1,136,158,379).  visible(ms3ggt_1,379). happensAt(walking(ms3ggt_1),379).
orientation(ms3ggt_0,158,380). coords(ms3ggt_0,141,142,380).  visible(ms3ggt_0,380). happensAt(active(ms3ggt_0),380). orientation(ms3ggt_1,162,380). coords(ms3ggt_1,138,158,380).  visible(ms3ggt_1,380). happensAt(walking(ms3ggt_1),380).
orientation(ms3ggt_0,152,381). coords(ms3ggt_0,141,142,381).  visible(ms3ggt_0,381). happensAt(active(ms3ggt_0),381). orientation(ms3ggt_1,162,381). coords(ms3ggt_1,138,158,381).  visible(ms3ggt_1,381). happensAt(walking(ms3ggt_1),381).
orientation(ms3ggt_0,152,382). coords(ms3ggt_0,142,141,382).  visible(ms3ggt_0,382). happensAt(active(ms3ggt_0),382). orientation(ms3ggt_1,162,382). coords(ms3ggt_1,140,158,382).  visible(ms3ggt_1,382). happensAt(walking(ms3ggt_1),382).
orientation(ms3ggt_0,152,383). coords(ms3ggt_0,142,141,383).  visible(ms3ggt_0,383). happensAt(active(ms3ggt_0),383). orientation(ms3ggt_1,162,383). coords(ms3ggt_1,141,158,383).  visible(ms3ggt_1,383). happensAt(walking(ms3ggt_1),383).
orientation(ms3ggt_0,152,384). coords(ms3ggt_0,142,140,384).  visible(ms3ggt_0,384). happensAt(active(ms3ggt_0),384). orientation(ms3ggt_1,162,384). coords(ms3ggt_1,142,158,384).  visible(ms3ggt_1,384). happensAt(walking(ms3ggt_1),384).
orientation(ms3ggt_0,152,385). coords(ms3ggt_0,142,140,385).  visible(ms3ggt_0,385). happensAt(active(ms3ggt_0),385). orientation(ms3ggt_1,162,385). coords(ms3ggt_1,142,157,385).  visible(ms3ggt_1,385). happensAt(walking(ms3ggt_1),385).
orientation(ms3ggt_0,152,386). coords(ms3ggt_0,142,140,386).  visible(ms3ggt_0,386). happensAt(active(ms3ggt_0),386). orientation(ms3ggt_1,162,386). coords(ms3ggt_1,142,157,386).  visible(ms3ggt_1,386). happensAt(walking(ms3ggt_1),386).
orientation(ms3ggt_0,152,387). coords(ms3ggt_0,142,140,387).  visible(ms3ggt_0,387). happensAt(active(ms3ggt_0),387). orientation(ms3ggt_1,162,387). coords(ms3ggt_1,142,157,387).  visible(ms3ggt_1,387). happensAt(walking(ms3ggt_1),387).
orientation(ms3ggt_0,152,388). coords(ms3ggt_0,142,140,388).  visible(ms3ggt_0,388). happensAt(active(ms3ggt_0),388). orientation(ms3ggt_1,162,388). coords(ms3ggt_1,143,157,388).  visible(ms3ggt_1,388). happensAt(walking(ms3ggt_1),388).
orientation(ms3ggt_0,152,389). coords(ms3ggt_0,142,139,389).  visible(ms3ggt_0,389). happensAt(active(ms3ggt_0),389). orientation(ms3ggt_1,162,389). coords(ms3ggt_1,143,157,389).  visible(ms3ggt_1,389). happensAt(walking(ms3ggt_1),389).
orientation(ms3ggt_0,152,390). coords(ms3ggt_0,142,139,390).  visible(ms3ggt_0,390). happensAt(active(ms3ggt_0),390). orientation(ms3ggt_1,162,390). coords(ms3ggt_1,143,157,390).  visible(ms3ggt_1,390). happensAt(walking(ms3ggt_1),390).
orientation(ms3ggt_0,152,391). coords(ms3ggt_0,142,138,391).  visible(ms3ggt_0,391). happensAt(active(ms3ggt_0),391). orientation(ms3ggt_1,162,391). coords(ms3ggt_1,144,157,391).  visible(ms3ggt_1,391). happensAt(walking(ms3ggt_1),391).
orientation(ms3ggt_0,152,392). coords(ms3ggt_0,142,138,392).  visible(ms3ggt_0,392). happensAt(active(ms3ggt_0),392). orientation(ms3ggt_1,162,392). coords(ms3ggt_1,144,158,392).  visible(ms3ggt_1,392). happensAt(walking(ms3ggt_1),392).
orientation(ms3ggt_0,152,393). coords(ms3ggt_0,142,138,393).  visible(ms3ggt_0,393). happensAt(active(ms3ggt_0),393). orientation(ms3ggt_1,153,393). coords(ms3ggt_1,144,158,393).  visible(ms3ggt_1,393). happensAt(walking(ms3ggt_1),393).
orientation(ms3ggt_0,152,394). coords(ms3ggt_0,141,138,394).  visible(ms3ggt_0,394). happensAt(active(ms3ggt_0),394). orientation(ms3ggt_1,153,394). coords(ms3ggt_1,147,158,394).  visible(ms3ggt_1,394). happensAt(walking(ms3ggt_1),394).
orientation(ms3ggt_0,152,395). coords(ms3ggt_0,141,138,395).  visible(ms3ggt_0,395). happensAt(walking(ms3ggt_0),395). orientation(ms3ggt_1,153,395). coords(ms3ggt_1,147,158,395).  visible(ms3ggt_1,395). happensAt(walking(ms3ggt_1),395).
orientation(ms3ggt_0,158,396). coords(ms3ggt_0,141,138,396).  visible(ms3ggt_0,396). happensAt(walking(ms3ggt_0),396). orientation(ms3ggt_1,153,396). coords(ms3ggt_1,149,158,396).  visible(ms3ggt_1,396). happensAt(walking(ms3ggt_1),396).
orientation(ms3ggt_0,158,397). coords(ms3ggt_0,140,138,397).  visible(ms3ggt_0,397). happensAt(walking(ms3ggt_0),397). orientation(ms3ggt_1,153,397). coords(ms3ggt_1,149,158,397).  visible(ms3ggt_1,397). happensAt(walking(ms3ggt_1),397).
orientation(ms3ggt_0,158,398). coords(ms3ggt_0,140,138,398).  visible(ms3ggt_0,398). happensAt(walking(ms3ggt_0),398). orientation(ms3ggt_1,153,398). coords(ms3ggt_1,149,158,398).  visible(ms3ggt_1,398). happensAt(walking(ms3ggt_1),398).
#show holdsAt/2.