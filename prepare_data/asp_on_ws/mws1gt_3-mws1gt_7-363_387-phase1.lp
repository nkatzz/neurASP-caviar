
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

	orientation(mws1gt_3,165,363). coords(mws1gt_3,49,159,363).  visible(mws1gt_3,363). happensAt(walking(mws1gt_3),363). orientation(mws1gt_7,174,363). coords(mws1gt_7,50,185,363).  visible(mws1gt_7,363). happensAt(walking(mws1gt_7),363).
orientation(mws1gt_3,165,364). coords(mws1gt_3,48,159,364).  visible(mws1gt_3,364). happensAt(walking(mws1gt_3),364). orientation(mws1gt_7,174,364). coords(mws1gt_7,50,185,364).  visible(mws1gt_7,364). happensAt(walking(mws1gt_7),364).
orientation(mws1gt_3,165,365). coords(mws1gt_3,47,159,365).  visible(mws1gt_3,365). happensAt(walking(mws1gt_3),365). orientation(mws1gt_7,174,365). coords(mws1gt_7,50,185,365).  visible(mws1gt_7,365). happensAt(walking(mws1gt_7),365).
orientation(mws1gt_3,165,366). coords(mws1gt_3,47,159,366).  visible(mws1gt_3,366). happensAt(walking(mws1gt_3),366). orientation(mws1gt_7,174,366). coords(mws1gt_7,50,185,366).  visible(mws1gt_7,366). happensAt(walking(mws1gt_7),366).
orientation(mws1gt_3,165,367). coords(mws1gt_3,47,159,367).  visible(mws1gt_3,367). happensAt(walking(mws1gt_3),367). orientation(mws1gt_7,174,367). coords(mws1gt_7,50,185,367).  visible(mws1gt_7,367). happensAt(walking(mws1gt_7),367).
orientation(mws1gt_3,165,368). coords(mws1gt_3,46,159,368).  visible(mws1gt_3,368). happensAt(walking(mws1gt_3),368). orientation(mws1gt_7,174,368). coords(mws1gt_7,50,185,368).  visible(mws1gt_7,368). happensAt(walking(mws1gt_7),368).
orientation(mws1gt_3,165,369). coords(mws1gt_3,45,160,369).  visible(mws1gt_3,369). happensAt(walking(mws1gt_3),369). orientation(mws1gt_7,174,369). coords(mws1gt_7,50,185,369).  visible(mws1gt_7,369). happensAt(walking(mws1gt_7),369).
orientation(mws1gt_3,165,370). coords(mws1gt_3,45,160,370).  visible(mws1gt_3,370). happensAt(walking(mws1gt_3),370). orientation(mws1gt_7,174,370). coords(mws1gt_7,50,185,370).  visible(mws1gt_7,370). happensAt(walking(mws1gt_7),370).
orientation(mws1gt_3,165,371). coords(mws1gt_3,44,160,371).  visible(mws1gt_3,371). happensAt(walking(mws1gt_3),371). orientation(mws1gt_7,174,371). coords(mws1gt_7,51,185,371).  visible(mws1gt_7,371). happensAt(walking(mws1gt_7),371).
orientation(mws1gt_3,165,372). coords(mws1gt_3,44,160,372).  visible(mws1gt_3,372). happensAt(walking(mws1gt_3),372). orientation(mws1gt_7,174,372). coords(mws1gt_7,53,185,372).  visible(mws1gt_7,372). happensAt(walking(mws1gt_7),372).
orientation(mws1gt_3,165,373). coords(mws1gt_3,44,160,373).  visible(mws1gt_3,373). happensAt(walking(mws1gt_3),373). orientation(mws1gt_7,174,373). coords(mws1gt_7,53,185,373).  visible(mws1gt_7,373). happensAt(walking(mws1gt_7),373).
orientation(mws1gt_3,165,374). coords(mws1gt_3,43,160,374).  visible(mws1gt_3,374). happensAt(walking(mws1gt_3),374). orientation(mws1gt_7,174,374). coords(mws1gt_7,55,185,374).  visible(mws1gt_7,374). happensAt(walking(mws1gt_7),374).
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
#show holdsAt/2.