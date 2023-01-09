
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

	orientation(fra2gt_3,138,358). coords(fra2gt_3,83,114,358).  visible(fra2gt_3,358). happensAt(walking(fra2gt_3),358). orientation(fra2gt_4,155,358). coords(fra2gt_4,95,117,358).  visible(fra2gt_4,358). happensAt(walking(fra2gt_4),358).
orientation(fra2gt_3,138,359). coords(fra2gt_3,83,115,359).  visible(fra2gt_3,359). happensAt(walking(fra2gt_3),359). orientation(fra2gt_4,155,359). coords(fra2gt_4,94,118,359).  visible(fra2gt_4,359). happensAt(walking(fra2gt_4),359).
orientation(fra2gt_3,138,360). coords(fra2gt_3,83,116,360).  visible(fra2gt_3,360). happensAt(walking(fra2gt_3),360). orientation(fra2gt_4,155,360). coords(fra2gt_4,94,118,360).  visible(fra2gt_4,360). happensAt(walking(fra2gt_4),360).
orientation(fra2gt_3,138,361). coords(fra2gt_3,83,117,361).  visible(fra2gt_3,361). happensAt(walking(fra2gt_3),361). orientation(fra2gt_4,155,361). coords(fra2gt_4,93,117,361).  visible(fra2gt_4,361). happensAt(walking(fra2gt_4),361).
orientation(fra2gt_3,138,362). coords(fra2gt_3,82,118,362).  visible(fra2gt_3,362). happensAt(walking(fra2gt_3),362). orientation(fra2gt_4,155,362). coords(fra2gt_4,92,118,362).  visible(fra2gt_4,362). happensAt(walking(fra2gt_4),362).
orientation(fra2gt_3,138,363). coords(fra2gt_3,82,118,363).  visible(fra2gt_3,363). happensAt(walking(fra2gt_3),363). orientation(fra2gt_4,155,363). coords(fra2gt_4,91,119,363).  visible(fra2gt_4,363). happensAt(walking(fra2gt_4),363).
orientation(fra2gt_3,138,364). coords(fra2gt_3,82,118,364).  visible(fra2gt_3,364). happensAt(walking(fra2gt_3),364). orientation(fra2gt_4,155,364). coords(fra2gt_4,90,119,364).  visible(fra2gt_4,364). happensAt(walking(fra2gt_4),364).
orientation(fra2gt_3,138,365). coords(fra2gt_3,76,118,365).  visible(fra2gt_3,365). happensAt(walking(fra2gt_3),365). orientation(fra2gt_4,155,365). coords(fra2gt_4,89,119,365).  visible(fra2gt_4,365). happensAt(walking(fra2gt_4),365).
orientation(fra2gt_3,138,366). coords(fra2gt_3,79,118,366).  visible(fra2gt_3,366). happensAt(walking(fra2gt_3),366). orientation(fra2gt_4,155,366). coords(fra2gt_4,87,117,366).  visible(fra2gt_4,366). happensAt(walking(fra2gt_4),366).
orientation(fra2gt_3,138,367). coords(fra2gt_3,79,120,367).  visible(fra2gt_3,367). happensAt(walking(fra2gt_3),367). orientation(fra2gt_4,155,367). coords(fra2gt_4,86,117,367).  visible(fra2gt_4,367). happensAt(walking(fra2gt_4),367).
orientation(fra2gt_3,138,368). coords(fra2gt_3,77,120,368).  visible(fra2gt_3,368). happensAt(walking(fra2gt_3),368). orientation(fra2gt_4,155,368). coords(fra2gt_4,84,117,368).  visible(fra2gt_4,368). happensAt(walking(fra2gt_4),368).
orientation(fra2gt_3,138,369). coords(fra2gt_3,77,121,369).  visible(fra2gt_3,369). happensAt(walking(fra2gt_3),369). orientation(fra2gt_4,155,369). coords(fra2gt_4,83,119,369).  visible(fra2gt_4,369). happensAt(walking(fra2gt_4),369).
orientation(fra2gt_3,138,370). coords(fra2gt_3,77,121,370).  visible(fra2gt_3,370). happensAt(walking(fra2gt_3),370). orientation(fra2gt_4,155,370). coords(fra2gt_4,83,120,370).  visible(fra2gt_4,370). happensAt(walking(fra2gt_4),370).
orientation(fra2gt_3,138,371). coords(fra2gt_3,76,121,371).  visible(fra2gt_3,371). happensAt(walking(fra2gt_3),371). orientation(fra2gt_4,155,371). coords(fra2gt_4,81,123,371).  visible(fra2gt_4,371). happensAt(walking(fra2gt_4),371).
orientation(fra2gt_3,138,372). coords(fra2gt_3,76,122,372).  visible(fra2gt_3,372). happensAt(walking(fra2gt_3),372). orientation(fra2gt_4,155,372). coords(fra2gt_4,81,124,372).  visible(fra2gt_4,372). happensAt(walking(fra2gt_4),372).
orientation(fra2gt_3,138,373). coords(fra2gt_3,76,122,373).  visible(fra2gt_3,373). happensAt(walking(fra2gt_3),373). orientation(fra2gt_4,155,373). coords(fra2gt_4,81,125,373).  visible(fra2gt_4,373). happensAt(walking(fra2gt_4),373).
orientation(fra2gt_3,138,374). coords(fra2gt_3,76,122,374).  visible(fra2gt_3,374). happensAt(walking(fra2gt_3),374). orientation(fra2gt_4,155,374). coords(fra2gt_4,81,125,374).  visible(fra2gt_4,374). happensAt(walking(fra2gt_4),374).
orientation(fra2gt_3,138,375). coords(fra2gt_3,76,124,375).  visible(fra2gt_3,375). happensAt(walking(fra2gt_3),375). orientation(fra2gt_4,155,375). coords(fra2gt_4,82,126,375).  visible(fra2gt_4,375). happensAt(walking(fra2gt_4),375).
orientation(fra2gt_3,138,376). coords(fra2gt_3,76,124,376).  visible(fra2gt_3,376). happensAt(walking(fra2gt_3),376). orientation(fra2gt_4,155,376). coords(fra2gt_4,82,126,376).  visible(fra2gt_4,376). happensAt(walking(fra2gt_4),376).
orientation(fra2gt_3,138,377). coords(fra2gt_3,75,125,377).  visible(fra2gt_3,377). happensAt(walking(fra2gt_3),377). orientation(fra2gt_4,153,377). coords(fra2gt_4,80,127,377).  visible(fra2gt_4,377). happensAt(walking(fra2gt_4),377).
orientation(fra2gt_3,138,378). coords(fra2gt_3,75,125,378).  visible(fra2gt_3,378). happensAt(walking(fra2gt_3),378). orientation(fra2gt_4,163,378). coords(fra2gt_4,79,130,378).  visible(fra2gt_4,378). happensAt(walking(fra2gt_4),378).
orientation(fra2gt_3,138,379). coords(fra2gt_3,75,125,379).  visible(fra2gt_3,379). happensAt(walking(fra2gt_3),379). orientation(fra2gt_4,163,379). coords(fra2gt_4,79,130,379).  visible(fra2gt_4,379). happensAt(walking(fra2gt_4),379).
orientation(fra2gt_3,138,380). coords(fra2gt_3,75,127,380).  visible(fra2gt_3,380). happensAt(walking(fra2gt_3),380). orientation(fra2gt_4,158,380). coords(fra2gt_4,78,131,380).  visible(fra2gt_4,380). happensAt(walking(fra2gt_4),380).
orientation(fra2gt_3,138,381). coords(fra2gt_3,74,127,381).  visible(fra2gt_3,381). happensAt(walking(fra2gt_3),381). orientation(fra2gt_4,150,381). coords(fra2gt_4,77,131,381).  visible(fra2gt_4,381). happensAt(walking(fra2gt_4),381).
orientation(fra2gt_3,138,382). coords(fra2gt_3,74,129,382).  visible(fra2gt_3,382). happensAt(walking(fra2gt_3),382). orientation(fra2gt_4,149,382). coords(fra2gt_4,78,132,382).  visible(fra2gt_4,382). happensAt(walking(fra2gt_4),382).
#show holdsAt/2.