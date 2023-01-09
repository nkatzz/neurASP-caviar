
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

	orientation(ms3ggt_0,158,344). coords(ms3ggt_0,112,144,344).  visible(ms3ggt_0,344). happensAt(walking(ms3ggt_0),344). orientation(ms3ggt_2,13,344). coords(ms3ggt_2,204,250,344).  visible(ms3ggt_2,344). happensAt(walking(ms3ggt_2),344).
orientation(ms3ggt_0,152,345). coords(ms3ggt_0,114,144,345).  visible(ms3ggt_0,345). happensAt(walking(ms3ggt_0),345). orientation(ms3ggt_2,13,345). coords(ms3ggt_2,204,250,345).  visible(ms3ggt_2,345). happensAt(walking(ms3ggt_2),345).
orientation(ms3ggt_0,152,346). coords(ms3ggt_0,116,144,346).  visible(ms3ggt_0,346). happensAt(walking(ms3ggt_0),346). orientation(ms3ggt_2,13,346). coords(ms3ggt_2,200,249,346).  visible(ms3ggt_2,346). happensAt(walking(ms3ggt_2),346).
orientation(ms3ggt_0,152,347). coords(ms3ggt_0,117,144,347).  visible(ms3ggt_0,347). happensAt(walking(ms3ggt_0),347). orientation(ms3ggt_2,8,347). coords(ms3ggt_2,198,244,347).  visible(ms3ggt_2,347). happensAt(walking(ms3ggt_2),347).
orientation(ms3ggt_0,157,348). coords(ms3ggt_0,118,144,348).  visible(ms3ggt_0,348). happensAt(walking(ms3ggt_0),348). orientation(ms3ggt_2,8,348). coords(ms3ggt_2,197,241,348).  visible(ms3ggt_2,348). happensAt(walking(ms3ggt_2),348).
orientation(ms3ggt_0,158,349). coords(ms3ggt_0,119,144,349).  visible(ms3ggt_0,349). happensAt(walking(ms3ggt_0),349). orientation(ms3ggt_2,8,349). coords(ms3ggt_2,197,240,349).  visible(ms3ggt_2,349). happensAt(walking(ms3ggt_2),349).
orientation(ms3ggt_0,158,350). coords(ms3ggt_0,120,145,350).  visible(ms3ggt_0,350). happensAt(walking(ms3ggt_0),350). orientation(ms3ggt_2,8,350). coords(ms3ggt_2,197,238,350).  visible(ms3ggt_2,350). happensAt(walking(ms3ggt_2),350).
orientation(ms3ggt_0,160,351). coords(ms3ggt_0,121,145,351).  visible(ms3ggt_0,351). happensAt(walking(ms3ggt_0),351). orientation(ms3ggt_2,8,351). coords(ms3ggt_2,196,238,351).  visible(ms3ggt_2,351). happensAt(walking(ms3ggt_2),351).
orientation(ms3ggt_0,160,352). coords(ms3ggt_0,121,145,352).  visible(ms3ggt_0,352). happensAt(walking(ms3ggt_0),352). orientation(ms3ggt_2,8,352). coords(ms3ggt_2,196,238,352).  visible(ms3ggt_2,352). happensAt(walking(ms3ggt_2),352).
orientation(ms3ggt_0,160,353). coords(ms3ggt_0,121,145,353).  visible(ms3ggt_0,353). happensAt(walking(ms3ggt_0),353). orientation(ms3ggt_2,8,353). coords(ms3ggt_2,195,237,353).  visible(ms3ggt_2,353). happensAt(walking(ms3ggt_2),353).
orientation(ms3ggt_0,160,354). coords(ms3ggt_0,121,145,354).  visible(ms3ggt_0,354). happensAt(walking(ms3ggt_0),354). orientation(ms3ggt_2,8,354). coords(ms3ggt_2,194,236,354).  visible(ms3ggt_2,354). happensAt(walking(ms3ggt_2),354).
orientation(ms3ggt_0,157,355). coords(ms3ggt_0,121,144,355).  visible(ms3ggt_0,355). happensAt(walking(ms3ggt_0),355). orientation(ms3ggt_2,8,355). coords(ms3ggt_2,191,234,355).  visible(ms3ggt_2,355). happensAt(walking(ms3ggt_2),355).
orientation(ms3ggt_0,157,356). coords(ms3ggt_0,122,144,356).  visible(ms3ggt_0,356). happensAt(walking(ms3ggt_0),356). orientation(ms3ggt_2,8,356). coords(ms3ggt_2,191,233,356).  visible(ms3ggt_2,356). happensAt(walking(ms3ggt_2),356).
orientation(ms3ggt_0,157,357). coords(ms3ggt_0,123,144,357).  visible(ms3ggt_0,357). happensAt(walking(ms3ggt_0),357). orientation(ms3ggt_2,176,357). coords(ms3ggt_2,190,233,357).  visible(ms3ggt_2,357). happensAt(walking(ms3ggt_2),357).
orientation(ms3ggt_0,157,358). coords(ms3ggt_0,125,144,358).  visible(ms3ggt_0,358). happensAt(walking(ms3ggt_0),358). orientation(ms3ggt_2,176,358). coords(ms3ggt_2,189,230,358).  visible(ms3ggt_2,358). happensAt(walking(ms3ggt_2),358).
orientation(ms3ggt_0,157,359). coords(ms3ggt_0,127,144,359).  visible(ms3ggt_0,359). happensAt(walking(ms3ggt_0),359). orientation(ms3ggt_2,176,359). coords(ms3ggt_2,188,227,359).  visible(ms3ggt_2,359). happensAt(walking(ms3ggt_2),359).
orientation(ms3ggt_0,161,360). coords(ms3ggt_0,129,144,360).  visible(ms3ggt_0,360). happensAt(walking(ms3ggt_0),360). orientation(ms3ggt_2,176,360). coords(ms3ggt_2,186,225,360).  visible(ms3ggt_2,360). happensAt(walking(ms3ggt_2),360).
orientation(ms3ggt_0,161,361). coords(ms3ggt_0,130,143,361).  visible(ms3ggt_0,361). happensAt(walking(ms3ggt_0),361). orientation(ms3ggt_2,176,361). coords(ms3ggt_2,186,223,361).  visible(ms3ggt_2,361). happensAt(walking(ms3ggt_2),361).
orientation(ms3ggt_0,161,362). coords(ms3ggt_0,131,143,362).  visible(ms3ggt_0,362). happensAt(walking(ms3ggt_0),362). orientation(ms3ggt_2,176,362). coords(ms3ggt_2,186,222,362).  visible(ms3ggt_2,362). happensAt(walking(ms3ggt_2),362).
orientation(ms3ggt_0,161,363). coords(ms3ggt_0,133,143,363).  visible(ms3ggt_0,363). happensAt(walking(ms3ggt_0),363). orientation(ms3ggt_2,173,363). coords(ms3ggt_2,185,221,363).  visible(ms3ggt_2,363). happensAt(walking(ms3ggt_2),363).
orientation(ms3ggt_0,158,364). coords(ms3ggt_0,133,143,364).  visible(ms3ggt_0,364). happensAt(walking(ms3ggt_0),364). orientation(ms3ggt_2,173,364). coords(ms3ggt_2,185,221,364).  visible(ms3ggt_2,364). happensAt(walking(ms3ggt_2),364).
orientation(ms3ggt_0,158,365). coords(ms3ggt_0,133,142,365).  visible(ms3ggt_0,365). happensAt(walking(ms3ggt_0),365). orientation(ms3ggt_2,173,365). coords(ms3ggt_2,184,220,365).  visible(ms3ggt_2,365). happensAt(walking(ms3ggt_2),365).
orientation(ms3ggt_0,158,366). coords(ms3ggt_0,134,142,366).  visible(ms3ggt_0,366). happensAt(walking(ms3ggt_0),366). orientation(ms3ggt_2,173,366). coords(ms3ggt_2,183,220,366).  visible(ms3ggt_2,366). happensAt(walking(ms3ggt_2),366).
orientation(ms3ggt_0,158,367). coords(ms3ggt_0,134,142,367).  visible(ms3ggt_0,367). happensAt(walking(ms3ggt_0),367). orientation(ms3ggt_2,173,367). coords(ms3ggt_2,183,218,367).  visible(ms3ggt_2,367). happensAt(walking(ms3ggt_2),367).
orientation(ms3ggt_0,158,368). coords(ms3ggt_0,134,142,368).  visible(ms3ggt_0,368). happensAt(walking(ms3ggt_0),368). orientation(ms3ggt_2,173,368). coords(ms3ggt_2,180,218,368).  visible(ms3ggt_2,368). happensAt(walking(ms3ggt_2),368).
#show holdsAt/2.