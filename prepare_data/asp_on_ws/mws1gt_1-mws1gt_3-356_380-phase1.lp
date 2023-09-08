
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

	orientation(mws1gt_1,143,356). coords(mws1gt_1,69,90,356).  visible(mws1gt_1,356). happensAt(walking(mws1gt_1),356). orientation(mws1gt_3,165,356). coords(mws1gt_3,51,155,356).  visible(mws1gt_3,356). happensAt(walking(mws1gt_3),356).
orientation(mws1gt_1,143,357). coords(mws1gt_1,69,89,357).  visible(mws1gt_1,357). happensAt(walking(mws1gt_1),357). orientation(mws1gt_3,165,357). coords(mws1gt_3,50,155,357).  visible(mws1gt_3,357). happensAt(walking(mws1gt_3),357).
orientation(mws1gt_1,143,358). coords(mws1gt_1,69,88,358).  visible(mws1gt_1,358). happensAt(walking(mws1gt_1),358). orientation(mws1gt_3,165,358). coords(mws1gt_3,50,157,358).  visible(mws1gt_3,358). happensAt(walking(mws1gt_3),358).
orientation(mws1gt_1,143,359). coords(mws1gt_1,69,88,359).  visible(mws1gt_1,359). happensAt(walking(mws1gt_1),359). orientation(mws1gt_3,165,359). coords(mws1gt_3,49,157,359).  visible(mws1gt_3,359). happensAt(walking(mws1gt_3),359).
orientation(mws1gt_1,143,360). coords(mws1gt_1,69,87,360).  visible(mws1gt_1,360). happensAt(walking(mws1gt_1),360). orientation(mws1gt_3,165,360). coords(mws1gt_3,49,158,360).  visible(mws1gt_3,360). happensAt(walking(mws1gt_3),360).
orientation(mws1gt_1,143,361). coords(mws1gt_1,69,87,361).  visible(mws1gt_1,361). happensAt(walking(mws1gt_1),361). orientation(mws1gt_3,165,361). coords(mws1gt_3,49,159,361).  visible(mws1gt_3,361). happensAt(walking(mws1gt_3),361).
orientation(mws1gt_1,143,362). coords(mws1gt_1,69,87,362).  visible(mws1gt_1,362). happensAt(walking(mws1gt_1),362). orientation(mws1gt_3,165,362). coords(mws1gt_3,49,159,362).  visible(mws1gt_3,362). happensAt(walking(mws1gt_3),362).
orientation(mws1gt_1,143,363). coords(mws1gt_1,69,86,363).  visible(mws1gt_1,363). happensAt(walking(mws1gt_1),363). orientation(mws1gt_3,165,363). coords(mws1gt_3,49,159,363).  visible(mws1gt_3,363). happensAt(walking(mws1gt_3),363).
orientation(mws1gt_1,143,364). coords(mws1gt_1,69,86,364).  visible(mws1gt_1,364). happensAt(walking(mws1gt_1),364). orientation(mws1gt_3,165,364). coords(mws1gt_3,48,159,364).  visible(mws1gt_3,364). happensAt(walking(mws1gt_3),364).
orientation(mws1gt_1,143,365). coords(mws1gt_1,70,85,365).  visible(mws1gt_1,365). happensAt(walking(mws1gt_1),365). orientation(mws1gt_3,165,365). coords(mws1gt_3,47,159,365).  visible(mws1gt_3,365). happensAt(walking(mws1gt_3),365).
orientation(mws1gt_1,143,366). coords(mws1gt_1,70,85,366).  visible(mws1gt_1,366). happensAt(walking(mws1gt_1),366). orientation(mws1gt_3,165,366). coords(mws1gt_3,47,159,366).  visible(mws1gt_3,366). happensAt(walking(mws1gt_3),366).
orientation(mws1gt_1,143,367). coords(mws1gt_1,70,84,367).  visible(mws1gt_1,367). happensAt(walking(mws1gt_1),367). orientation(mws1gt_3,165,367). coords(mws1gt_3,47,159,367).  visible(mws1gt_3,367). happensAt(walking(mws1gt_3),367).
orientation(mws1gt_1,143,368). coords(mws1gt_1,70,84,368).  visible(mws1gt_1,368). happensAt(walking(mws1gt_1),368). orientation(mws1gt_3,165,368). coords(mws1gt_3,46,159,368).  visible(mws1gt_3,368). happensAt(walking(mws1gt_3),368).
orientation(mws1gt_1,143,369). coords(mws1gt_1,70,84,369).  visible(mws1gt_1,369). happensAt(walking(mws1gt_1),369). orientation(mws1gt_3,165,369). coords(mws1gt_3,45,160,369).  visible(mws1gt_3,369). happensAt(walking(mws1gt_3),369).
orientation(mws1gt_1,143,370). coords(mws1gt_1,70,84,370).  visible(mws1gt_1,370). happensAt(walking(mws1gt_1),370). orientation(mws1gt_3,165,370). coords(mws1gt_3,45,160,370).  visible(mws1gt_3,370). happensAt(walking(mws1gt_3),370).
orientation(mws1gt_1,143,371). coords(mws1gt_1,70,82,371).  visible(mws1gt_1,371). happensAt(walking(mws1gt_1),371). orientation(mws1gt_3,165,371). coords(mws1gt_3,44,160,371).  visible(mws1gt_3,371). happensAt(walking(mws1gt_3),371).
orientation(mws1gt_1,143,372). coords(mws1gt_1,70,81,372).  visible(mws1gt_1,372). happensAt(walking(mws1gt_1),372). orientation(mws1gt_3,165,372). coords(mws1gt_3,44,160,372).  visible(mws1gt_3,372). happensAt(walking(mws1gt_3),372).
orientation(mws1gt_1,143,373). coords(mws1gt_1,70,80,373).  visible(mws1gt_1,373). happensAt(walking(mws1gt_1),373). orientation(mws1gt_3,165,373). coords(mws1gt_3,44,160,373).  visible(mws1gt_3,373). happensAt(walking(mws1gt_3),373).
orientation(mws1gt_1,143,374). coords(mws1gt_1,70,79,374).  visible(mws1gt_1,374). happensAt(walking(mws1gt_1),374). orientation(mws1gt_3,165,374). coords(mws1gt_3,43,160,374).  visible(mws1gt_3,374). happensAt(walking(mws1gt_3),374).
orientation(mws1gt_1,143,375). coords(mws1gt_1,70,78,375).  visible(mws1gt_1,375). happensAt(walking(mws1gt_1),375). orientation(mws1gt_3,165,375). coords(mws1gt_3,42,160,375).  visible(mws1gt_3,375). happensAt(walking(mws1gt_3),375).
orientation(mws1gt_1,143,376). coords(mws1gt_1,70,77,376).  visible(mws1gt_1,376). happensAt(walking(mws1gt_1),376). orientation(mws1gt_3,165,376). coords(mws1gt_3,42,160,376).  visible(mws1gt_3,376). happensAt(walking(mws1gt_3),376).
orientation(mws1gt_1,143,377). coords(mws1gt_1,70,77,377).  visible(mws1gt_1,377). happensAt(walking(mws1gt_1),377). orientation(mws1gt_3,165,377). coords(mws1gt_3,42,160,377).  visible(mws1gt_3,377). happensAt(walking(mws1gt_3),377).
orientation(mws1gt_1,143,378). coords(mws1gt_1,70,76,378).  visible(mws1gt_1,378). happensAt(walking(mws1gt_1),378). orientation(mws1gt_3,165,378). coords(mws1gt_3,42,160,378).  visible(mws1gt_3,378). happensAt(walking(mws1gt_3),378).
orientation(mws1gt_1,143,379). coords(mws1gt_1,70,76,379).  visible(mws1gt_1,379). happensAt(walking(mws1gt_1),379). orientation(mws1gt_3,165,379). coords(mws1gt_3,42,160,379).  visible(mws1gt_3,379). happensAt(walking(mws1gt_3),379).
orientation(mws1gt_1,143,380). coords(mws1gt_1,70,76,380).  visible(mws1gt_1,380). happensAt(walking(mws1gt_1),380). orientation(mws1gt_3,165,380). coords(mws1gt_3,42,160,380).  visible(mws1gt_3,380). happensAt(walking(mws1gt_3),380).
#show holdsAt/2.