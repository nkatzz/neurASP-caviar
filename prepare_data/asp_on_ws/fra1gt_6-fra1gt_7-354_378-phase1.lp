
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

	orientation(fra1gt_6,134,354). coords(fra1gt_6,159,86,354).  visible(fra1gt_6,354). happensAt(walking(fra1gt_6),354). orientation(fra1gt_7,155,354). coords(fra1gt_7,176,68,354).  visible(fra1gt_7,354). happensAt(walking(fra1gt_7),354).
orientation(fra1gt_6,134,355). coords(fra1gt_6,160,87,355).  visible(fra1gt_6,355). happensAt(walking(fra1gt_6),355). orientation(fra1gt_7,155,355). coords(fra1gt_7,175,69,355).  visible(fra1gt_7,355). happensAt(walking(fra1gt_7),355).
orientation(fra1gt_6,134,356). coords(fra1gt_6,160,87,356).  visible(fra1gt_6,356). happensAt(walking(fra1gt_6),356). orientation(fra1gt_7,160,356). coords(fra1gt_7,175,69,356).  visible(fra1gt_7,356). happensAt(walking(fra1gt_7),356).
orientation(fra1gt_6,134,357). coords(fra1gt_6,159,87,357).  visible(fra1gt_6,357). happensAt(walking(fra1gt_6),357). orientation(fra1gt_7,160,357). coords(fra1gt_7,173,69,357).  visible(fra1gt_7,357). happensAt(walking(fra1gt_7),357).
orientation(fra1gt_6,134,358). coords(fra1gt_6,158,87,358).  visible(fra1gt_6,358). happensAt(walking(fra1gt_6),358). orientation(fra1gt_7,160,358). coords(fra1gt_7,173,69,358).  visible(fra1gt_7,358). happensAt(walking(fra1gt_7),358).
orientation(fra1gt_6,134,359). coords(fra1gt_6,159,86,359).  visible(fra1gt_6,359). happensAt(walking(fra1gt_6),359). orientation(fra1gt_7,160,359). coords(fra1gt_7,174,69,359).  visible(fra1gt_7,359). happensAt(walking(fra1gt_7),359).
orientation(fra1gt_6,134,360). coords(fra1gt_6,158,86,360).  visible(fra1gt_6,360). happensAt(walking(fra1gt_6),360). orientation(fra1gt_7,160,360). coords(fra1gt_7,174,69,360).  visible(fra1gt_7,360). happensAt(walking(fra1gt_7),360).
orientation(fra1gt_6,134,361). coords(fra1gt_6,156,86,361).  visible(fra1gt_6,361). happensAt(walking(fra1gt_6),361). orientation(fra1gt_7,160,361). coords(fra1gt_7,175,69,361).  visible(fra1gt_7,361). happensAt(walking(fra1gt_7),361).
orientation(fra1gt_6,134,362). coords(fra1gt_6,155,86,362).  visible(fra1gt_6,362). happensAt(walking(fra1gt_6),362). orientation(fra1gt_7,160,362). coords(fra1gt_7,174,70,362).  visible(fra1gt_7,362). happensAt(walking(fra1gt_7),362).
orientation(fra1gt_6,126,363). coords(fra1gt_6,155,86,363).  visible(fra1gt_6,363). happensAt(walking(fra1gt_6),363). orientation(fra1gt_7,160,363). coords(fra1gt_7,173,70,363).  visible(fra1gt_7,363). happensAt(walking(fra1gt_7),363).
orientation(fra1gt_6,126,364). coords(fra1gt_6,155,87,364).  visible(fra1gt_6,364). happensAt(walking(fra1gt_6),364). orientation(fra1gt_7,160,364). coords(fra1gt_7,173,70,364).  visible(fra1gt_7,364). happensAt(walking(fra1gt_7),364).
orientation(fra1gt_6,126,365). coords(fra1gt_6,155,88,365).  visible(fra1gt_6,365). happensAt(walking(fra1gt_6),365). orientation(fra1gt_7,160,365). coords(fra1gt_7,173,70,365).  visible(fra1gt_7,365). happensAt(walking(fra1gt_7),365).
orientation(fra1gt_6,126,366). coords(fra1gt_6,155,88,366).  visible(fra1gt_6,366). happensAt(walking(fra1gt_6),366). orientation(fra1gt_7,160,366). coords(fra1gt_7,173,70,366).  visible(fra1gt_7,366). happensAt(walking(fra1gt_7),366).
orientation(fra1gt_6,126,367). coords(fra1gt_6,155,88,367).  visible(fra1gt_6,367). happensAt(walking(fra1gt_6),367). orientation(fra1gt_7,160,367). coords(fra1gt_7,173,70,367).  visible(fra1gt_7,367). happensAt(walking(fra1gt_7),367).
orientation(fra1gt_6,126,368). coords(fra1gt_6,154,88,368).  visible(fra1gt_6,368). happensAt(walking(fra1gt_6),368). orientation(fra1gt_7,160,368). coords(fra1gt_7,173,70,368).  visible(fra1gt_7,368). happensAt(walking(fra1gt_7),368).
orientation(fra1gt_6,126,369). coords(fra1gt_6,153,88,369).  visible(fra1gt_6,369). happensAt(walking(fra1gt_6),369). orientation(fra1gt_7,162,369). coords(fra1gt_7,173,71,369).  visible(fra1gt_7,369). happensAt(walking(fra1gt_7),369).
orientation(fra1gt_6,126,370). coords(fra1gt_6,153,88,370).  visible(fra1gt_6,370). happensAt(walking(fra1gt_6),370). orientation(fra1gt_7,162,370). coords(fra1gt_7,173,71,370).  visible(fra1gt_7,370). happensAt(walking(fra1gt_7),370).
orientation(fra1gt_6,126,371). coords(fra1gt_6,152,87,371).  visible(fra1gt_6,371). happensAt(walking(fra1gt_6),371). orientation(fra1gt_7,162,371). coords(fra1gt_7,172,69,371).  visible(fra1gt_7,371). happensAt(walking(fra1gt_7),371).
orientation(fra1gt_6,126,372). coords(fra1gt_6,152,88,372).  visible(fra1gt_6,372). happensAt(walking(fra1gt_6),372). orientation(fra1gt_7,162,372). coords(fra1gt_7,169,69,372).  visible(fra1gt_7,372). happensAt(walking(fra1gt_7),372).
orientation(fra1gt_6,136,373). coords(fra1gt_6,155,88,373).  visible(fra1gt_6,373). happensAt(walking(fra1gt_6),373). orientation(fra1gt_7,162,373). coords(fra1gt_7,167,69,373).  visible(fra1gt_7,373). happensAt(running(fra1gt_7),373).
orientation(fra1gt_6,136,374). coords(fra1gt_6,155,89,374).  visible(fra1gt_6,374). happensAt(walking(fra1gt_6),374). orientation(fra1gt_7,162,374). coords(fra1gt_7,164,69,374).  visible(fra1gt_7,374). happensAt(running(fra1gt_7),374).
orientation(fra1gt_6,136,375). coords(fra1gt_6,155,90,375).  visible(fra1gt_6,375). happensAt(walking(fra1gt_6),375). orientation(fra1gt_7,162,375). coords(fra1gt_7,164,70,375).  visible(fra1gt_7,375). happensAt(running(fra1gt_7),375).
orientation(fra1gt_6,136,376). coords(fra1gt_6,155,90,376).  visible(fra1gt_6,376). happensAt(walking(fra1gt_6),376). orientation(fra1gt_7,162,376). coords(fra1gt_7,162,71,376).  visible(fra1gt_7,376). happensAt(running(fra1gt_7),376).
orientation(fra1gt_6,136,377). coords(fra1gt_6,155,92,377).  visible(fra1gt_6,377). happensAt(walking(fra1gt_6),377). orientation(fra1gt_7,162,377). coords(fra1gt_7,159,71,377).  visible(fra1gt_7,377). happensAt(running(fra1gt_7),377).
orientation(fra1gt_6,136,378). coords(fra1gt_6,156,93,378).  visible(fra1gt_6,378). happensAt(walking(fra1gt_6),378). orientation(fra1gt_7,162,378). coords(fra1gt_7,157,70,378).  visible(fra1gt_7,378). happensAt(running(fra1gt_7),378).
#show holdsAt/2.