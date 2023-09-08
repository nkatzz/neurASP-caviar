
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

	orientation(wk1gt_1,149,356). coords(wk1gt_1,136,130,356).  visible(wk1gt_1,356). happensAt(walking(wk1gt_1),356). orientation(wk1gt_4,158,356). coords(wk1gt_4,52,176,356).  visible(wk1gt_4,356). happensAt(inactive(wk1gt_4),356).
orientation(wk1gt_1,145,357). coords(wk1gt_1,133,128,357).  visible(wk1gt_1,357). happensAt(walking(wk1gt_1),357). orientation(wk1gt_4,157,357). coords(wk1gt_4,52,176,357).  visible(wk1gt_4,357). happensAt(inactive(wk1gt_4),357).
orientation(wk1gt_1,145,358). coords(wk1gt_1,131,126,358).  visible(wk1gt_1,358). happensAt(walking(wk1gt_1),358). orientation(wk1gt_4,161,358). coords(wk1gt_4,52,176,358).  visible(wk1gt_4,358). happensAt(inactive(wk1gt_4),358).
orientation(wk1gt_1,145,359). coords(wk1gt_1,131,126,359).  visible(wk1gt_1,359). happensAt(walking(wk1gt_1),359). orientation(wk1gt_4,161,359). coords(wk1gt_4,52,176,359).  visible(wk1gt_4,359). happensAt(inactive(wk1gt_4),359).
orientation(wk1gt_1,151,360). coords(wk1gt_1,130,126,360).  visible(wk1gt_1,360). happensAt(walking(wk1gt_1),360). orientation(wk1gt_4,162,360). coords(wk1gt_4,52,176,360).  visible(wk1gt_4,360). happensAt(inactive(wk1gt_4),360).
orientation(wk1gt_1,151,361). coords(wk1gt_1,129,124,361).  visible(wk1gt_1,361). happensAt(walking(wk1gt_1),361). orientation(wk1gt_4,163,361). coords(wk1gt_4,52,176,361).  visible(wk1gt_4,361). happensAt(inactive(wk1gt_4),361).
orientation(wk1gt_1,151,362). coords(wk1gt_1,129,124,362).  visible(wk1gt_1,362). happensAt(walking(wk1gt_1),362). orientation(wk1gt_4,168,362). coords(wk1gt_4,52,176,362).  visible(wk1gt_4,362). happensAt(inactive(wk1gt_4),362).
orientation(wk1gt_1,148,363). coords(wk1gt_1,129,124,363).  visible(wk1gt_1,363). happensAt(walking(wk1gt_1),363). orientation(wk1gt_4,164,363). coords(wk1gt_4,52,176,363).  visible(wk1gt_4,363). happensAt(inactive(wk1gt_4),363).
orientation(wk1gt_1,148,364). coords(wk1gt_1,129,123,364).  visible(wk1gt_1,364). happensAt(walking(wk1gt_1),364). orientation(wk1gt_4,166,364). coords(wk1gt_4,52,176,364).  visible(wk1gt_4,364). happensAt(inactive(wk1gt_4),364).
orientation(wk1gt_1,148,365). coords(wk1gt_1,128,123,365).  visible(wk1gt_1,365). happensAt(walking(wk1gt_1),365). orientation(wk1gt_4,162,365). coords(wk1gt_4,52,176,365).  visible(wk1gt_4,365). happensAt(inactive(wk1gt_4),365).
orientation(wk1gt_1,148,366). coords(wk1gt_1,128,123,366).  visible(wk1gt_1,366). happensAt(walking(wk1gt_1),366). orientation(wk1gt_4,164,366). coords(wk1gt_4,52,176,366).  visible(wk1gt_4,366). happensAt(inactive(wk1gt_4),366).
orientation(wk1gt_1,153,367). coords(wk1gt_1,127,122,367).  visible(wk1gt_1,367). happensAt(walking(wk1gt_1),367). orientation(wk1gt_4,162,367). coords(wk1gt_4,52,176,367).  visible(wk1gt_4,367). happensAt(inactive(wk1gt_4),367).
orientation(wk1gt_1,153,368). coords(wk1gt_1,126,121,368).  visible(wk1gt_1,368). happensAt(walking(wk1gt_1),368). orientation(wk1gt_4,161,368). coords(wk1gt_4,52,176,368).  visible(wk1gt_4,368). happensAt(inactive(wk1gt_4),368).
orientation(wk1gt_1,153,369). coords(wk1gt_1,124,120,369).  visible(wk1gt_1,369). happensAt(walking(wk1gt_1),369). orientation(wk1gt_4,162,369). coords(wk1gt_4,52,176,369).  visible(wk1gt_4,369). happensAt(inactive(wk1gt_4),369).
orientation(wk1gt_1,153,370). coords(wk1gt_1,124,120,370).  visible(wk1gt_1,370). happensAt(walking(wk1gt_1),370). orientation(wk1gt_4,163,370). coords(wk1gt_4,52,176,370).  visible(wk1gt_4,370). happensAt(inactive(wk1gt_4),370).
orientation(wk1gt_1,153,371). coords(wk1gt_1,123,119,371).  visible(wk1gt_1,371). happensAt(walking(wk1gt_1),371). orientation(wk1gt_4,164,371). coords(wk1gt_4,52,176,371).  visible(wk1gt_4,371). happensAt(inactive(wk1gt_4),371).
orientation(wk1gt_1,153,372). coords(wk1gt_1,120,119,372).  visible(wk1gt_1,372). happensAt(walking(wk1gt_1),372). orientation(wk1gt_4,163,372). coords(wk1gt_4,52,176,372).  visible(wk1gt_4,372). happensAt(inactive(wk1gt_4),372).
orientation(wk1gt_1,149,373). coords(wk1gt_1,119,119,373).  visible(wk1gt_1,373). happensAt(walking(wk1gt_1),373). orientation(wk1gt_4,164,373). coords(wk1gt_4,52,176,373).  visible(wk1gt_4,373). happensAt(inactive(wk1gt_4),373).
orientation(wk1gt_1,149,374). coords(wk1gt_1,119,119,374).  visible(wk1gt_1,374). happensAt(walking(wk1gt_1),374). orientation(wk1gt_4,166,374). coords(wk1gt_4,52,176,374).  visible(wk1gt_4,374). happensAt(inactive(wk1gt_4),374).
orientation(wk1gt_1,149,375). coords(wk1gt_1,118,118,375).  visible(wk1gt_1,375). happensAt(walking(wk1gt_1),375). orientation(wk1gt_4,170,375). coords(wk1gt_4,52,176,375).  visible(wk1gt_4,375). happensAt(inactive(wk1gt_4),375).
orientation(wk1gt_1,149,376). coords(wk1gt_1,118,118,376).  visible(wk1gt_1,376). happensAt(walking(wk1gt_1),376). orientation(wk1gt_4,163,376). coords(wk1gt_4,52,176,376).  visible(wk1gt_4,376). happensAt(inactive(wk1gt_4),376).
orientation(wk1gt_1,149,377). coords(wk1gt_1,118,118,377).  visible(wk1gt_1,377). happensAt(walking(wk1gt_1),377). orientation(wk1gt_4,163,377). coords(wk1gt_4,52,176,377).  visible(wk1gt_4,377). happensAt(inactive(wk1gt_4),377).
orientation(wk1gt_1,149,378). coords(wk1gt_1,118,118,378).  visible(wk1gt_1,378). happensAt(walking(wk1gt_1),378). orientation(wk1gt_4,161,378). coords(wk1gt_4,52,176,378).  visible(wk1gt_4,378). happensAt(inactive(wk1gt_4),378).
orientation(wk1gt_1,149,379). coords(wk1gt_1,118,118,379).  visible(wk1gt_1,379). happensAt(walking(wk1gt_1),379). orientation(wk1gt_4,154,379). coords(wk1gt_4,52,176,379).  visible(wk1gt_4,379). happensAt(inactive(wk1gt_4),379).
orientation(wk1gt_1,147,380). coords(wk1gt_1,117,117,380).  visible(wk1gt_1,380). happensAt(walking(wk1gt_1),380). orientation(wk1gt_4,157,380). coords(wk1gt_4,52,176,380).  visible(wk1gt_4,380). happensAt(inactive(wk1gt_4),380).
#show holdsAt/2.