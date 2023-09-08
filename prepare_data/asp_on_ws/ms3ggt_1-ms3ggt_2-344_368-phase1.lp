
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

	orientation(ms3ggt_1,156,344). coords(ms3ggt_1,106,166,344).  visible(ms3ggt_1,344). happensAt(walking(ms3ggt_1),344). orientation(ms3ggt_2,13,344). coords(ms3ggt_2,204,250,344).  visible(ms3ggt_2,344). happensAt(walking(ms3ggt_2),344).
orientation(ms3ggt_1,156,345). coords(ms3ggt_1,108,165,345).  visible(ms3ggt_1,345). happensAt(walking(ms3ggt_1),345). orientation(ms3ggt_2,13,345). coords(ms3ggt_2,204,250,345).  visible(ms3ggt_2,345). happensAt(walking(ms3ggt_2),345).
orientation(ms3ggt_1,156,346). coords(ms3ggt_1,110,165,346).  visible(ms3ggt_1,346). happensAt(walking(ms3ggt_1),346). orientation(ms3ggt_2,13,346). coords(ms3ggt_2,200,249,346).  visible(ms3ggt_2,346). happensAt(walking(ms3ggt_2),346).
orientation(ms3ggt_1,156,347). coords(ms3ggt_1,111,164,347).  visible(ms3ggt_1,347). happensAt(walking(ms3ggt_1),347). orientation(ms3ggt_2,8,347). coords(ms3ggt_2,198,244,347).  visible(ms3ggt_2,347). happensAt(walking(ms3ggt_2),347).
orientation(ms3ggt_1,156,348). coords(ms3ggt_1,112,164,348).  visible(ms3ggt_1,348). happensAt(walking(ms3ggt_1),348). orientation(ms3ggt_2,8,348). coords(ms3ggt_2,197,241,348).  visible(ms3ggt_2,348). happensAt(walking(ms3ggt_2),348).
orientation(ms3ggt_1,156,349). coords(ms3ggt_1,113,164,349).  visible(ms3ggt_1,349). happensAt(walking(ms3ggt_1),349). orientation(ms3ggt_2,8,349). coords(ms3ggt_2,197,240,349).  visible(ms3ggt_2,349). happensAt(walking(ms3ggt_2),349).
orientation(ms3ggt_1,156,350). coords(ms3ggt_1,114,164,350).  visible(ms3ggt_1,350). happensAt(walking(ms3ggt_1),350). orientation(ms3ggt_2,8,350). coords(ms3ggt_2,197,238,350).  visible(ms3ggt_2,350). happensAt(walking(ms3ggt_2),350).
orientation(ms3ggt_1,156,351). coords(ms3ggt_1,115,164,351).  visible(ms3ggt_1,351). happensAt(walking(ms3ggt_1),351). orientation(ms3ggt_2,8,351). coords(ms3ggt_2,196,238,351).  visible(ms3ggt_2,351). happensAt(walking(ms3ggt_2),351).
orientation(ms3ggt_1,156,352). coords(ms3ggt_1,115,163,352).  visible(ms3ggt_1,352). happensAt(walking(ms3ggt_1),352). orientation(ms3ggt_2,8,352). coords(ms3ggt_2,196,238,352).  visible(ms3ggt_2,352). happensAt(walking(ms3ggt_2),352).
orientation(ms3ggt_1,156,353). coords(ms3ggt_1,115,163,353).  visible(ms3ggt_1,353). happensAt(walking(ms3ggt_1),353). orientation(ms3ggt_2,8,353). coords(ms3ggt_2,195,237,353).  visible(ms3ggt_2,353). happensAt(walking(ms3ggt_2),353).
orientation(ms3ggt_1,156,354). coords(ms3ggt_1,115,163,354).  visible(ms3ggt_1,354). happensAt(walking(ms3ggt_1),354). orientation(ms3ggt_2,8,354). coords(ms3ggt_2,194,236,354).  visible(ms3ggt_2,354). happensAt(walking(ms3ggt_2),354).
orientation(ms3ggt_1,156,355). coords(ms3ggt_1,116,162,355).  visible(ms3ggt_1,355). happensAt(walking(ms3ggt_1),355). orientation(ms3ggt_2,8,355). coords(ms3ggt_2,191,234,355).  visible(ms3ggt_2,355). happensAt(walking(ms3ggt_2),355).
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
#show holdsAt/2.