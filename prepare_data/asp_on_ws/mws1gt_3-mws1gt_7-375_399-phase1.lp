
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

	orientation(mws1gt_3,165,375). coords(mws1gt_3,42,160,375).  visible(mws1gt_3,375). happensAt(walking(mws1gt_3),375). orientation(mws1gt_7,174,375). coords(mws1gt_7,56,184,375).  visible(mws1gt_7,375). happensAt(walking(mws1gt_7),375).
orientation(mws1gt_3,165,376). coords(mws1gt_3,42,160,376).  visible(mws1gt_3,376). happensAt(walking(mws1gt_3),376). orientation(mws1gt_7,174,376). coords(mws1gt_7,56,184,376).  visible(mws1gt_7,376). happensAt(walking(mws1gt_7),376).
orientation(mws1gt_3,165,377). coords(mws1gt_3,42,160,377).  visible(mws1gt_3,377). happensAt(walking(mws1gt_3),377). orientation(mws1gt_7,174,377). coords(mws1gt_7,56,184,377).  visible(mws1gt_7,377). happensAt(walking(mws1gt_7),377).
orientation(mws1gt_3,165,378). coords(mws1gt_3,42,160,378).  visible(mws1gt_3,378). happensAt(walking(mws1gt_3),378). orientation(mws1gt_7,174,378). coords(mws1gt_7,56,184,378).  visible(mws1gt_7,378). happensAt(walking(mws1gt_7),378).
orientation(mws1gt_3,165,379). coords(mws1gt_3,42,160,379).  visible(mws1gt_3,379). happensAt(walking(mws1gt_3),379). orientation(mws1gt_7,174,379). coords(mws1gt_7,56,184,379).  visible(mws1gt_7,379). happensAt(walking(mws1gt_7),379).
orientation(mws1gt_3,165,380). coords(mws1gt_3,42,160,380).  visible(mws1gt_3,380). happensAt(walking(mws1gt_3),380). orientation(mws1gt_7,174,380). coords(mws1gt_7,56,184,380).  visible(mws1gt_7,380). happensAt(walking(mws1gt_7),380).
orientation(mws1gt_3,165,381). coords(mws1gt_3,42,161,381).  visible(mws1gt_3,381). happensAt(walking(mws1gt_3),381). orientation(mws1gt_7,174,381). coords(mws1gt_7,56,184,381).  visible(mws1gt_7,381). happensAt(walking(mws1gt_7),381).
orientation(mws1gt_3,165,382). coords(mws1gt_3,41,161,382).  visible(mws1gt_3,382). happensAt(walking(mws1gt_3),382). orientation(mws1gt_7,174,382). coords(mws1gt_7,56,185,382).  visible(mws1gt_7,382). happensAt(walking(mws1gt_7),382).
orientation(mws1gt_3,165,383). coords(mws1gt_3,41,161,383).  visible(mws1gt_3,383). happensAt(walking(mws1gt_3),383). orientation(mws1gt_7,174,383). coords(mws1gt_7,56,185,383).  visible(mws1gt_7,383). happensAt(walking(mws1gt_7),383).
orientation(mws1gt_3,165,384). coords(mws1gt_3,40,161,384).  visible(mws1gt_3,384). happensAt(walking(mws1gt_3),384). orientation(mws1gt_7,174,384). coords(mws1gt_7,57,184,384).  visible(mws1gt_7,384). happensAt(walking(mws1gt_7),384).
orientation(mws1gt_3,165,385). coords(mws1gt_3,39,161,385).  visible(mws1gt_3,385). happensAt(walking(mws1gt_3),385). orientation(mws1gt_7,174,385). coords(mws1gt_7,57,184,385).  visible(mws1gt_7,385). happensAt(walking(mws1gt_7),385).
orientation(mws1gt_3,165,386). coords(mws1gt_3,39,161,386).  visible(mws1gt_3,386). happensAt(walking(mws1gt_3),386). orientation(mws1gt_7,174,386). coords(mws1gt_7,57,185,386).  visible(mws1gt_7,386). happensAt(walking(mws1gt_7),386).
orientation(mws1gt_3,165,387). coords(mws1gt_3,39,161,387).  visible(mws1gt_3,387). happensAt(walking(mws1gt_3),387). orientation(mws1gt_7,174,387). coords(mws1gt_7,58,185,387).  visible(mws1gt_7,387). happensAt(walking(mws1gt_7),387).
orientation(mws1gt_3,165,388). coords(mws1gt_3,39,161,388).  visible(mws1gt_3,388). happensAt(walking(mws1gt_3),388). orientation(mws1gt_7,174,388). coords(mws1gt_7,59,185,388).  visible(mws1gt_7,388). happensAt(walking(mws1gt_7),388).
orientation(mws1gt_3,165,389). coords(mws1gt_3,39,161,389).  visible(mws1gt_3,389). happensAt(walking(mws1gt_3),389). orientation(mws1gt_7,174,389). coords(mws1gt_7,60,186,389).  visible(mws1gt_7,389). happensAt(walking(mws1gt_7),389).
orientation(mws1gt_3,165,390). coords(mws1gt_3,38,161,390).  visible(mws1gt_3,390). happensAt(walking(mws1gt_3),390). orientation(mws1gt_7,174,390). coords(mws1gt_7,60,186,390).  visible(mws1gt_7,390). happensAt(walking(mws1gt_7),390).
orientation(mws1gt_3,165,391). coords(mws1gt_3,38,161,391).  visible(mws1gt_3,391). happensAt(walking(mws1gt_3),391). orientation(mws1gt_7,174,391). coords(mws1gt_7,61,187,391).  visible(mws1gt_7,391). happensAt(walking(mws1gt_7),391).
orientation(mws1gt_3,165,392). coords(mws1gt_3,38,161,392).  visible(mws1gt_3,392). happensAt(walking(mws1gt_3),392). orientation(mws1gt_7,174,392). coords(mws1gt_7,61,187,392).  visible(mws1gt_7,392). happensAt(walking(mws1gt_7),392).
orientation(mws1gt_3,165,393). coords(mws1gt_3,38,161,393).  visible(mws1gt_3,393). happensAt(walking(mws1gt_3),393). orientation(mws1gt_7,174,393). coords(mws1gt_7,61,187,393).  visible(mws1gt_7,393). happensAt(walking(mws1gt_7),393).
orientation(mws1gt_3,165,394). coords(mws1gt_3,38,161,394).  visible(mws1gt_3,394). happensAt(walking(mws1gt_3),394). orientation(mws1gt_7,174,394). coords(mws1gt_7,61,187,394).  visible(mws1gt_7,394). happensAt(walking(mws1gt_7),394).
orientation(mws1gt_3,165,395). coords(mws1gt_3,37,161,395).  visible(mws1gt_3,395). happensAt(walking(mws1gt_3),395). orientation(mws1gt_7,174,395). coords(mws1gt_7,61,187,395).  visible(mws1gt_7,395). happensAt(walking(mws1gt_7),395).
orientation(mws1gt_3,165,396). coords(mws1gt_3,37,161,396).  visible(mws1gt_3,396). happensAt(walking(mws1gt_3),396). orientation(mws1gt_7,174,396). coords(mws1gt_7,61,187,396).  visible(mws1gt_7,396). happensAt(walking(mws1gt_7),396).
orientation(mws1gt_3,165,397). coords(mws1gt_3,37,161,397).  visible(mws1gt_3,397). happensAt(walking(mws1gt_3),397). orientation(mws1gt_7,174,397). coords(mws1gt_7,61,187,397).  visible(mws1gt_7,397). happensAt(walking(mws1gt_7),397).
orientation(mws1gt_3,165,398). coords(mws1gt_3,37,161,398).  visible(mws1gt_3,398). happensAt(walking(mws1gt_3),398). orientation(mws1gt_7,174,398). coords(mws1gt_7,61,187,398).  visible(mws1gt_7,398). happensAt(walking(mws1gt_7),398).
orientation(mws1gt_3,165,399). coords(mws1gt_3,37,161,399).  visible(mws1gt_3,399). happensAt(walking(mws1gt_3),399). orientation(mws1gt_7,174,399). coords(mws1gt_7,62,188,399).  visible(mws1gt_7,399). happensAt(walking(mws1gt_7),399).
#show holdsAt/2.