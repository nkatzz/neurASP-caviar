
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

	orientation(mws1gt_1,143,375). coords(mws1gt_1,70,78,375).  visible(mws1gt_1,375). happensAt(walking(mws1gt_1),375). orientation(mws1gt_7,174,375). coords(mws1gt_7,56,184,375).  visible(mws1gt_7,375). happensAt(walking(mws1gt_7),375).
orientation(mws1gt_1,143,376). coords(mws1gt_1,70,77,376).  visible(mws1gt_1,376). happensAt(walking(mws1gt_1),376). orientation(mws1gt_7,174,376). coords(mws1gt_7,56,184,376).  visible(mws1gt_7,376). happensAt(walking(mws1gt_7),376).
orientation(mws1gt_1,143,377). coords(mws1gt_1,70,77,377).  visible(mws1gt_1,377). happensAt(walking(mws1gt_1),377). orientation(mws1gt_7,174,377). coords(mws1gt_7,56,184,377).  visible(mws1gt_7,377). happensAt(walking(mws1gt_7),377).
orientation(mws1gt_1,143,378). coords(mws1gt_1,70,76,378).  visible(mws1gt_1,378). happensAt(walking(mws1gt_1),378). orientation(mws1gt_7,174,378). coords(mws1gt_7,56,184,378).  visible(mws1gt_7,378). happensAt(walking(mws1gt_7),378).
orientation(mws1gt_1,143,379). coords(mws1gt_1,70,76,379).  visible(mws1gt_1,379). happensAt(walking(mws1gt_1),379). orientation(mws1gt_7,174,379). coords(mws1gt_7,56,184,379).  visible(mws1gt_7,379). happensAt(walking(mws1gt_7),379).
orientation(mws1gt_1,143,380). coords(mws1gt_1,70,76,380).  visible(mws1gt_1,380). happensAt(walking(mws1gt_1),380). orientation(mws1gt_7,174,380). coords(mws1gt_7,56,184,380).  visible(mws1gt_7,380). happensAt(walking(mws1gt_7),380).
orientation(mws1gt_1,143,381). coords(mws1gt_1,70,76,381).  visible(mws1gt_1,381). happensAt(walking(mws1gt_1),381). orientation(mws1gt_7,174,381). coords(mws1gt_7,56,184,381).  visible(mws1gt_7,381). happensAt(walking(mws1gt_7),381).
orientation(mws1gt_1,143,382). coords(mws1gt_1,71,76,382).  visible(mws1gt_1,382). happensAt(walking(mws1gt_1),382). orientation(mws1gt_7,174,382). coords(mws1gt_7,56,185,382).  visible(mws1gt_7,382). happensAt(walking(mws1gt_7),382).
orientation(mws1gt_1,143,383). coords(mws1gt_1,71,75,383).  visible(mws1gt_1,383). happensAt(walking(mws1gt_1),383). orientation(mws1gt_7,174,383). coords(mws1gt_7,56,185,383).  visible(mws1gt_7,383). happensAt(walking(mws1gt_7),383).
orientation(mws1gt_1,143,384). coords(mws1gt_1,71,75,384).  visible(mws1gt_1,384). happensAt(walking(mws1gt_1),384). orientation(mws1gt_7,174,384). coords(mws1gt_7,57,184,384).  visible(mws1gt_7,384). happensAt(walking(mws1gt_7),384).
orientation(mws1gt_1,143,385). coords(mws1gt_1,71,75,385).  visible(mws1gt_1,385). happensAt(walking(mws1gt_1),385). orientation(mws1gt_7,174,385). coords(mws1gt_7,57,184,385).  visible(mws1gt_7,385). happensAt(walking(mws1gt_7),385).
orientation(mws1gt_1,143,386). coords(mws1gt_1,71,74,386).  visible(mws1gt_1,386). happensAt(walking(mws1gt_1),386). orientation(mws1gt_7,174,386). coords(mws1gt_7,57,185,386).  visible(mws1gt_7,386). happensAt(walking(mws1gt_7),386).
orientation(mws1gt_1,143,387). coords(mws1gt_1,72,72,387).  visible(mws1gt_1,387). happensAt(walking(mws1gt_1),387). orientation(mws1gt_7,174,387). coords(mws1gt_7,58,185,387).  visible(mws1gt_7,387). happensAt(walking(mws1gt_7),387).
orientation(mws1gt_1,143,388). coords(mws1gt_1,72,71,388).  visible(mws1gt_1,388). happensAt(walking(mws1gt_1),388). orientation(mws1gt_7,174,388). coords(mws1gt_7,59,185,388).  visible(mws1gt_7,388). happensAt(walking(mws1gt_7),388).
orientation(mws1gt_1,143,389). coords(mws1gt_1,72,71,389).  visible(mws1gt_1,389). happensAt(walking(mws1gt_1),389). orientation(mws1gt_7,174,389). coords(mws1gt_7,60,186,389).  visible(mws1gt_7,389). happensAt(walking(mws1gt_7),389).
orientation(mws1gt_1,143,390). coords(mws1gt_1,72,70,390).  visible(mws1gt_1,390). happensAt(walking(mws1gt_1),390). orientation(mws1gt_7,174,390). coords(mws1gt_7,60,186,390).  visible(mws1gt_7,390). happensAt(walking(mws1gt_7),390).
orientation(mws1gt_1,143,391). coords(mws1gt_1,72,69,391).  visible(mws1gt_1,391). happensAt(walking(mws1gt_1),391). orientation(mws1gt_7,174,391). coords(mws1gt_7,61,187,391).  visible(mws1gt_7,391). happensAt(walking(mws1gt_7),391).
orientation(mws1gt_1,143,392). coords(mws1gt_1,72,68,392).  visible(mws1gt_1,392). happensAt(walking(mws1gt_1),392). orientation(mws1gt_7,174,392). coords(mws1gt_7,61,187,392).  visible(mws1gt_7,392). happensAt(walking(mws1gt_7),392).
orientation(mws1gt_1,143,393). coords(mws1gt_1,72,68,393).  visible(mws1gt_1,393). happensAt(walking(mws1gt_1),393). orientation(mws1gt_7,174,393). coords(mws1gt_7,61,187,393).  visible(mws1gt_7,393). happensAt(walking(mws1gt_7),393).
orientation(mws1gt_1,143,394). coords(mws1gt_1,72,67,394).  visible(mws1gt_1,394). happensAt(walking(mws1gt_1),394). orientation(mws1gt_7,174,394). coords(mws1gt_7,61,187,394).  visible(mws1gt_7,394). happensAt(walking(mws1gt_7),394).
orientation(mws1gt_1,143,395). coords(mws1gt_1,73,67,395).  visible(mws1gt_1,395). happensAt(walking(mws1gt_1),395). orientation(mws1gt_7,174,395). coords(mws1gt_7,61,187,395).  visible(mws1gt_7,395). happensAt(walking(mws1gt_7),395).
orientation(mws1gt_1,143,396). coords(mws1gt_1,73,67,396).  visible(mws1gt_1,396). happensAt(walking(mws1gt_1),396). orientation(mws1gt_7,174,396). coords(mws1gt_7,61,187,396).  visible(mws1gt_7,396). happensAt(walking(mws1gt_7),396).
orientation(mws1gt_1,143,397). coords(mws1gt_1,73,67,397).  visible(mws1gt_1,397). happensAt(walking(mws1gt_1),397). orientation(mws1gt_7,174,397). coords(mws1gt_7,61,187,397).  visible(mws1gt_7,397). happensAt(walking(mws1gt_7),397).
orientation(mws1gt_1,143,398). coords(mws1gt_1,73,67,398).  visible(mws1gt_1,398). happensAt(walking(mws1gt_1),398). orientation(mws1gt_7,174,398). coords(mws1gt_7,61,187,398).  visible(mws1gt_7,398). happensAt(walking(mws1gt_7),398).
orientation(mws1gt_1,143,399). coords(mws1gt_1,73,67,399).  visible(mws1gt_1,399). happensAt(walking(mws1gt_1),399). orientation(mws1gt_7,174,399). coords(mws1gt_7,62,188,399).  visible(mws1gt_7,399). happensAt(walking(mws1gt_7),399).
#show holdsAt/2.