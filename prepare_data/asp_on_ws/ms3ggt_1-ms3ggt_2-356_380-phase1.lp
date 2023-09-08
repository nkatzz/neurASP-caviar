
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

	orientation(ms3ggt_1,156,356). coords(ms3ggt_1,116,162,356).  visible(ms3ggt_1,356). happensAt(walking(ms3ggt_1),356). orientation(ms3ggt_2,8,356). coords(ms3ggt_2,191,233,356).  visible(ms3ggt_2,356). happensAt(walking(ms3ggt_2),356).
orientation(ms3ggt_1,156,357). coords(ms3ggt_1,117,161,357).  visible(ms3ggt_1,357). happensAt(walking(ms3ggt_1),357). orientation(ms3ggt_2,176,357). coords(ms3ggt_2,190,233,357).  visible(ms3ggt_2,357). happensAt(walking(ms3ggt_2),357).
orientation(ms3ggt_1,156,358). coords(ms3ggt_1,117,161,358).  visible(ms3ggt_1,358). happensAt(walking(ms3ggt_1),358). orientation(ms3ggt_2,176,358). coords(ms3ggt_2,189,230,358).  visible(ms3ggt_2,358). happensAt(walking(ms3ggt_2),358).
orientation(ms3ggt_1,156,359). coords(ms3ggt_1,117,161,359).  visible(ms3ggt_1,359). happensAt(walking(ms3ggt_1),359). orientation(ms3ggt_2,176,359). coords(ms3ggt_2,188,227,359).  visible(ms3ggt_2,359). happensAt(walking(ms3ggt_2),359).
orientation(ms3ggt_1,156,360). coords(ms3ggt_1,117,161,360).  visible(ms3ggt_1,360). happensAt(walking(ms3ggt_1),360). orientation(ms3ggt_2,176,360). coords(ms3ggt_2,186,225,360).  visible(ms3ggt_2,360). happensAt(walking(ms3ggt_2),360).
orientation(ms3ggt_1,156,361). coords(ms3ggt_1,117,159,361).  visible(ms3ggt_1,361). happensAt(walking(ms3ggt_1),361). orientation(ms3ggt_2,176,361). coords(ms3ggt_2,186,223,361).  visible(ms3ggt_2,361). happensAt(walking(ms3ggt_2),361).
orientation(ms3ggt_1,156,362). coords(ms3ggt_1,118,159,362).  visible(ms3ggt_1,362). happensAt(walking(ms3ggt_1),362). orientation(ms3ggt_2,176,362). coords(ms3ggt_2,186,222,362).  visible(ms3ggt_2,362). happensAt(walking(ms3ggt_2),362).
orientation(ms3ggt_1,156,363). coords(ms3ggt_1,121,159,363).  visible(ms3ggt_1,363). happensAt(walking(ms3ggt_1),363). orientation(ms3ggt_2,173,363). coords(ms3ggt_2,185,221,363).  visible(ms3ggt_2,363). happensAt(walking(ms3ggt_2),363).
orientation(ms3ggt_1,156,364). coords(ms3ggt_1,123,159,364).  visible(ms3ggt_1,364). happensAt(walking(ms3ggt_1),364). orientation(ms3ggt_2,173,364). coords(ms3ggt_2,185,221,364).  visible(ms3ggt_2,364). happensAt(walking(ms3ggt_2),364).
orientation(ms3ggt_1,156,365). coords(ms3ggt_1,124,159,365).  visible(ms3ggt_1,365). happensAt(walking(ms3ggt_1),365). orientation(ms3ggt_2,173,365). coords(ms3ggt_2,184,220,365).  visible(ms3ggt_2,365). happensAt(walking(ms3ggt_2),365).
orientation(ms3ggt_1,156,366). coords(ms3ggt_1,125,158,366).  visible(ms3ggt_1,366). happensAt(walking(ms3ggt_1),366). orientation(ms3ggt_2,173,366). coords(ms3ggt_2,183,220,366).  visible(ms3ggt_2,366). happensAt(walking(ms3ggt_2),366).
orientation(ms3ggt_1,156,367). coords(ms3ggt_1,125,158,367).  visible(ms3ggt_1,367). happensAt(walking(ms3ggt_1),367). orientation(ms3ggt_2,173,367). coords(ms3ggt_2,183,218,367).  visible(ms3ggt_2,367). happensAt(walking(ms3ggt_2),367).
orientation(ms3ggt_1,156,368). coords(ms3ggt_1,126,158,368).  visible(ms3ggt_1,368). happensAt(walking(ms3ggt_1),368). orientation(ms3ggt_2,173,368). coords(ms3ggt_2,180,218,368).  visible(ms3ggt_2,368). happensAt(walking(ms3ggt_2),368).
orientation(ms3ggt_1,156,369). coords(ms3ggt_1,127,158,369).  visible(ms3ggt_1,369). happensAt(walking(ms3ggt_1),369). orientation(ms3ggt_2,173,369). coords(ms3ggt_2,178,217,369).  visible(ms3ggt_2,369). happensAt(walking(ms3ggt_2),369).
orientation(ms3ggt_1,156,370). coords(ms3ggt_1,127,158,370).  visible(ms3ggt_1,370). happensAt(walking(ms3ggt_1),370). orientation(ms3ggt_2,173,370). coords(ms3ggt_2,175,217,370).  visible(ms3ggt_2,370). happensAt(walking(ms3ggt_2),370).
orientation(ms3ggt_1,156,371). coords(ms3ggt_1,128,158,371).  visible(ms3ggt_1,371). happensAt(walking(ms3ggt_1),371). orientation(ms3ggt_2,173,371). coords(ms3ggt_2,174,217,371).  visible(ms3ggt_2,371). happensAt(walking(ms3ggt_2),371).
orientation(ms3ggt_1,156,372). coords(ms3ggt_1,128,158,372).  visible(ms3ggt_1,372). happensAt(walking(ms3ggt_1),372). orientation(ms3ggt_2,173,372). coords(ms3ggt_2,171,216,372).  visible(ms3ggt_2,372). happensAt(walking(ms3ggt_2),372).
orientation(ms3ggt_1,156,373). coords(ms3ggt_1,129,158,373).  visible(ms3ggt_1,373). happensAt(walking(ms3ggt_1),373). orientation(ms3ggt_2,173,373). coords(ms3ggt_2,170,216,373).  visible(ms3ggt_2,373). happensAt(walking(ms3ggt_2),373).
orientation(ms3ggt_1,156,374). coords(ms3ggt_1,130,158,374).  visible(ms3ggt_1,374). happensAt(walking(ms3ggt_1),374). orientation(ms3ggt_2,173,374). coords(ms3ggt_2,169,215,374).  visible(ms3ggt_2,374). happensAt(walking(ms3ggt_2),374).
orientation(ms3ggt_1,156,375). coords(ms3ggt_1,130,158,375).  visible(ms3ggt_1,375). happensAt(walking(ms3ggt_1),375). orientation(ms3ggt_2,173,375). coords(ms3ggt_2,169,215,375).  visible(ms3ggt_2,375). happensAt(walking(ms3ggt_2),375).
orientation(ms3ggt_1,156,376). coords(ms3ggt_1,130,158,376).  visible(ms3ggt_1,376). happensAt(walking(ms3ggt_1),376). orientation(ms3ggt_2,173,376). coords(ms3ggt_2,168,212,376).  visible(ms3ggt_2,376). happensAt(walking(ms3ggt_2),376).
orientation(ms3ggt_1,156,377). coords(ms3ggt_1,132,158,377).  visible(ms3ggt_1,377). happensAt(walking(ms3ggt_1),377). orientation(ms3ggt_2,173,377). coords(ms3ggt_2,168,212,377).  visible(ms3ggt_2,377). happensAt(walking(ms3ggt_2),377).
orientation(ms3ggt_1,156,378). coords(ms3ggt_1,134,158,378).  visible(ms3ggt_1,378). happensAt(walking(ms3ggt_1),378). orientation(ms3ggt_2,173,378). coords(ms3ggt_2,167,211,378).  visible(ms3ggt_2,378). happensAt(walking(ms3ggt_2),378).
orientation(ms3ggt_1,162,379). coords(ms3ggt_1,136,158,379).  visible(ms3ggt_1,379). happensAt(walking(ms3ggt_1),379). orientation(ms3ggt_2,173,379). coords(ms3ggt_2,166,210,379).  visible(ms3ggt_2,379). happensAt(walking(ms3ggt_2),379).
orientation(ms3ggt_1,162,380). coords(ms3ggt_1,138,158,380).  visible(ms3ggt_1,380). happensAt(walking(ms3ggt_1),380). orientation(ms3ggt_2,173,380). coords(ms3ggt_2,166,210,380).  visible(ms3ggt_2,380). happensAt(walking(ms3ggt_2),380).
#show holdsAt/2.