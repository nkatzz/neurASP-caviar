
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

	orientation(mws1gt_1,143,363). coords(mws1gt_1,69,86,363).  visible(mws1gt_1,363). happensAt(walking(mws1gt_1),363). orientation(mws1gt_7,174,363). coords(mws1gt_7,50,185,363).  visible(mws1gt_7,363). happensAt(walking(mws1gt_7),363).
orientation(mws1gt_1,143,364). coords(mws1gt_1,69,86,364).  visible(mws1gt_1,364). happensAt(walking(mws1gt_1),364). orientation(mws1gt_7,174,364). coords(mws1gt_7,50,185,364).  visible(mws1gt_7,364). happensAt(walking(mws1gt_7),364).
orientation(mws1gt_1,143,365). coords(mws1gt_1,70,85,365).  visible(mws1gt_1,365). happensAt(walking(mws1gt_1),365). orientation(mws1gt_7,174,365). coords(mws1gt_7,50,185,365).  visible(mws1gt_7,365). happensAt(walking(mws1gt_7),365).
orientation(mws1gt_1,143,366). coords(mws1gt_1,70,85,366).  visible(mws1gt_1,366). happensAt(walking(mws1gt_1),366). orientation(mws1gt_7,174,366). coords(mws1gt_7,50,185,366).  visible(mws1gt_7,366). happensAt(walking(mws1gt_7),366).
orientation(mws1gt_1,143,367). coords(mws1gt_1,70,84,367).  visible(mws1gt_1,367). happensAt(walking(mws1gt_1),367). orientation(mws1gt_7,174,367). coords(mws1gt_7,50,185,367).  visible(mws1gt_7,367). happensAt(walking(mws1gt_7),367).
orientation(mws1gt_1,143,368). coords(mws1gt_1,70,84,368).  visible(mws1gt_1,368). happensAt(walking(mws1gt_1),368). orientation(mws1gt_7,174,368). coords(mws1gt_7,50,185,368).  visible(mws1gt_7,368). happensAt(walking(mws1gt_7),368).
orientation(mws1gt_1,143,369). coords(mws1gt_1,70,84,369).  visible(mws1gt_1,369). happensAt(walking(mws1gt_1),369). orientation(mws1gt_7,174,369). coords(mws1gt_7,50,185,369).  visible(mws1gt_7,369). happensAt(walking(mws1gt_7),369).
orientation(mws1gt_1,143,370). coords(mws1gt_1,70,84,370).  visible(mws1gt_1,370). happensAt(walking(mws1gt_1),370). orientation(mws1gt_7,174,370). coords(mws1gt_7,50,185,370).  visible(mws1gt_7,370). happensAt(walking(mws1gt_7),370).
orientation(mws1gt_1,143,371). coords(mws1gt_1,70,82,371).  visible(mws1gt_1,371). happensAt(walking(mws1gt_1),371). orientation(mws1gt_7,174,371). coords(mws1gt_7,51,185,371).  visible(mws1gt_7,371). happensAt(walking(mws1gt_7),371).
orientation(mws1gt_1,143,372). coords(mws1gt_1,70,81,372).  visible(mws1gt_1,372). happensAt(walking(mws1gt_1),372). orientation(mws1gt_7,174,372). coords(mws1gt_7,53,185,372).  visible(mws1gt_7,372). happensAt(walking(mws1gt_7),372).
orientation(mws1gt_1,143,373). coords(mws1gt_1,70,80,373).  visible(mws1gt_1,373). happensAt(walking(mws1gt_1),373). orientation(mws1gt_7,174,373). coords(mws1gt_7,53,185,373).  visible(mws1gt_7,373). happensAt(walking(mws1gt_7),373).
orientation(mws1gt_1,143,374). coords(mws1gt_1,70,79,374).  visible(mws1gt_1,374). happensAt(walking(mws1gt_1),374). orientation(mws1gt_7,174,374). coords(mws1gt_7,55,185,374).  visible(mws1gt_7,374). happensAt(walking(mws1gt_7),374).
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
#show holdsAt/2.