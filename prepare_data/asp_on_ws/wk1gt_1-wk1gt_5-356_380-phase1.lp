
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

	orientation(wk1gt_1,149,356). coords(wk1gt_1,136,130,356).  visible(wk1gt_1,356). happensAt(walking(wk1gt_1),356). orientation(wk1gt_5,162,356). coords(wk1gt_5,50,163,356).  visible(wk1gt_5,356). happensAt(inactive(wk1gt_5),356).
orientation(wk1gt_1,145,357). coords(wk1gt_1,133,128,357).  visible(wk1gt_1,357). happensAt(walking(wk1gt_1),357). orientation(wk1gt_5,162,357). coords(wk1gt_5,50,163,357).  visible(wk1gt_5,357). happensAt(inactive(wk1gt_5),357).
orientation(wk1gt_1,145,358). coords(wk1gt_1,131,126,358).  visible(wk1gt_1,358). happensAt(walking(wk1gt_1),358). orientation(wk1gt_5,162,358). coords(wk1gt_5,50,163,358).  visible(wk1gt_5,358). happensAt(inactive(wk1gt_5),358).
orientation(wk1gt_1,145,359). coords(wk1gt_1,131,126,359).  visible(wk1gt_1,359). happensAt(walking(wk1gt_1),359). orientation(wk1gt_5,162,359). coords(wk1gt_5,50,163,359).  visible(wk1gt_5,359). happensAt(inactive(wk1gt_5),359).
orientation(wk1gt_1,151,360). coords(wk1gt_1,130,126,360).  visible(wk1gt_1,360). happensAt(walking(wk1gt_1),360). orientation(wk1gt_5,162,360). coords(wk1gt_5,50,163,360).  visible(wk1gt_5,360). happensAt(inactive(wk1gt_5),360).
orientation(wk1gt_1,151,361). coords(wk1gt_1,129,124,361).  visible(wk1gt_1,361). happensAt(walking(wk1gt_1),361). orientation(wk1gt_5,162,361). coords(wk1gt_5,50,163,361).  visible(wk1gt_5,361). happensAt(inactive(wk1gt_5),361).
orientation(wk1gt_1,151,362). coords(wk1gt_1,129,124,362).  visible(wk1gt_1,362). happensAt(walking(wk1gt_1),362). orientation(wk1gt_5,162,362). coords(wk1gt_5,50,163,362).  visible(wk1gt_5,362). happensAt(inactive(wk1gt_5),362).
orientation(wk1gt_1,148,363). coords(wk1gt_1,129,124,363).  visible(wk1gt_1,363). happensAt(walking(wk1gt_1),363). orientation(wk1gt_5,162,363). coords(wk1gt_5,50,163,363).  visible(wk1gt_5,363). happensAt(inactive(wk1gt_5),363).
orientation(wk1gt_1,148,364). coords(wk1gt_1,129,123,364).  visible(wk1gt_1,364). happensAt(walking(wk1gt_1),364). orientation(wk1gt_5,162,364). coords(wk1gt_5,51,163,364).  visible(wk1gt_5,364). happensAt(inactive(wk1gt_5),364).
orientation(wk1gt_1,148,365). coords(wk1gt_1,128,123,365).  visible(wk1gt_1,365). happensAt(walking(wk1gt_1),365). orientation(wk1gt_5,162,365). coords(wk1gt_5,50,163,365).  visible(wk1gt_5,365). happensAt(inactive(wk1gt_5),365).
orientation(wk1gt_1,148,366). coords(wk1gt_1,128,123,366).  visible(wk1gt_1,366). happensAt(walking(wk1gt_1),366). orientation(wk1gt_5,162,366). coords(wk1gt_5,50,163,366).  visible(wk1gt_5,366). happensAt(inactive(wk1gt_5),366).
orientation(wk1gt_1,153,367). coords(wk1gt_1,127,122,367).  visible(wk1gt_1,367). happensAt(walking(wk1gt_1),367). orientation(wk1gt_5,162,367). coords(wk1gt_5,50,163,367).  visible(wk1gt_5,367). happensAt(inactive(wk1gt_5),367).
orientation(wk1gt_1,153,368). coords(wk1gt_1,126,121,368).  visible(wk1gt_1,368). happensAt(walking(wk1gt_1),368). orientation(wk1gt_5,162,368). coords(wk1gt_5,51,163,368).  visible(wk1gt_5,368). happensAt(inactive(wk1gt_5),368).
orientation(wk1gt_1,153,369). coords(wk1gt_1,124,120,369).  visible(wk1gt_1,369). happensAt(walking(wk1gt_1),369). orientation(wk1gt_5,162,369). coords(wk1gt_5,50,163,369).  visible(wk1gt_5,369). happensAt(inactive(wk1gt_5),369).
orientation(wk1gt_1,153,370). coords(wk1gt_1,124,120,370).  visible(wk1gt_1,370). happensAt(walking(wk1gt_1),370). orientation(wk1gt_5,162,370). coords(wk1gt_5,50,163,370).  visible(wk1gt_5,370). happensAt(inactive(wk1gt_5),370).
orientation(wk1gt_1,153,371). coords(wk1gt_1,123,119,371).  visible(wk1gt_1,371). happensAt(walking(wk1gt_1),371). orientation(wk1gt_5,162,371). coords(wk1gt_5,50,163,371).  visible(wk1gt_5,371). happensAt(inactive(wk1gt_5),371).
orientation(wk1gt_1,153,372). coords(wk1gt_1,120,119,372).  visible(wk1gt_1,372). happensAt(walking(wk1gt_1),372). orientation(wk1gt_5,162,372). coords(wk1gt_5,50,163,372).  visible(wk1gt_5,372). happensAt(inactive(wk1gt_5),372).
orientation(wk1gt_1,149,373). coords(wk1gt_1,119,119,373).  visible(wk1gt_1,373). happensAt(walking(wk1gt_1),373). orientation(wk1gt_5,162,373). coords(wk1gt_5,50,163,373).  visible(wk1gt_5,373). happensAt(inactive(wk1gt_5),373).
orientation(wk1gt_1,149,374). coords(wk1gt_1,119,119,374).  visible(wk1gt_1,374). happensAt(walking(wk1gt_1),374). orientation(wk1gt_5,162,374). coords(wk1gt_5,50,163,374).  visible(wk1gt_5,374). happensAt(inactive(wk1gt_5),374).
orientation(wk1gt_1,149,375). coords(wk1gt_1,118,118,375).  visible(wk1gt_1,375). happensAt(walking(wk1gt_1),375). orientation(wk1gt_5,162,375). coords(wk1gt_5,50,163,375).  visible(wk1gt_5,375). happensAt(inactive(wk1gt_5),375).
orientation(wk1gt_1,149,376). coords(wk1gt_1,118,118,376).  visible(wk1gt_1,376). happensAt(walking(wk1gt_1),376). orientation(wk1gt_5,162,376). coords(wk1gt_5,50,163,376).  visible(wk1gt_5,376). happensAt(inactive(wk1gt_5),376).
orientation(wk1gt_1,149,377). coords(wk1gt_1,118,118,377).  visible(wk1gt_1,377). happensAt(walking(wk1gt_1),377). orientation(wk1gt_5,162,377). coords(wk1gt_5,51,163,377).  visible(wk1gt_5,377). happensAt(inactive(wk1gt_5),377).
orientation(wk1gt_1,149,378). coords(wk1gt_1,118,118,378).  visible(wk1gt_1,378). happensAt(walking(wk1gt_1),378). orientation(wk1gt_5,162,378). coords(wk1gt_5,50,163,378).  visible(wk1gt_5,378). happensAt(inactive(wk1gt_5),378).
orientation(wk1gt_1,149,379). coords(wk1gt_1,118,118,379).  visible(wk1gt_1,379). happensAt(walking(wk1gt_1),379). orientation(wk1gt_5,162,379). coords(wk1gt_5,50,163,379).  visible(wk1gt_5,379). happensAt(inactive(wk1gt_5),379).
orientation(wk1gt_1,147,380). coords(wk1gt_1,117,117,380).  visible(wk1gt_1,380). happensAt(walking(wk1gt_1),380). orientation(wk1gt_5,162,380). coords(wk1gt_5,50,163,380).  visible(wk1gt_5,380). happensAt(inactive(wk1gt_5),380).
#show holdsAt/2.