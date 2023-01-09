
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

	orientation(wk1gt_1,149,344). coords(wk1gt_1,147,137,344).  visible(wk1gt_1,344). happensAt(walking(wk1gt_1),344). orientation(wk1gt_4,167,344). coords(wk1gt_4,52,176,344).  visible(wk1gt_4,344). happensAt(inactive(wk1gt_4),344).
orientation(wk1gt_1,149,345). coords(wk1gt_1,145,136,345).  visible(wk1gt_1,345). happensAt(walking(wk1gt_1),345). orientation(wk1gt_4,164,345). coords(wk1gt_4,52,176,345).  visible(wk1gt_4,345). happensAt(inactive(wk1gt_4),345).
orientation(wk1gt_1,149,346). coords(wk1gt_1,143,134,346).  visible(wk1gt_1,346). happensAt(walking(wk1gt_1),346). orientation(wk1gt_4,168,346). coords(wk1gt_4,52,176,346).  visible(wk1gt_4,346). happensAt(inactive(wk1gt_4),346).
orientation(wk1gt_1,149,347). coords(wk1gt_1,141,133,347).  visible(wk1gt_1,347). happensAt(walking(wk1gt_1),347). orientation(wk1gt_4,168,347). coords(wk1gt_4,52,176,347).  visible(wk1gt_4,347). happensAt(inactive(wk1gt_4),347).
orientation(wk1gt_1,149,348). coords(wk1gt_1,140,133,348).  visible(wk1gt_1,348). happensAt(walking(wk1gt_1),348). orientation(wk1gt_4,166,348). coords(wk1gt_4,52,176,348).  visible(wk1gt_4,348). happensAt(inactive(wk1gt_4),348).
orientation(wk1gt_1,149,349). coords(wk1gt_1,139,133,349).  visible(wk1gt_1,349). happensAt(walking(wk1gt_1),349). orientation(wk1gt_4,165,349). coords(wk1gt_4,52,176,349).  visible(wk1gt_4,349). happensAt(inactive(wk1gt_4),349).
orientation(wk1gt_1,149,350). coords(wk1gt_1,139,132,350).  visible(wk1gt_1,350). happensAt(walking(wk1gt_1),350). orientation(wk1gt_4,165,350). coords(wk1gt_4,52,176,350).  visible(wk1gt_4,350). happensAt(inactive(wk1gt_4),350).
orientation(wk1gt_1,149,351). coords(wk1gt_1,139,132,351).  visible(wk1gt_1,351). happensAt(walking(wk1gt_1),351). orientation(wk1gt_4,167,351). coords(wk1gt_4,52,176,351).  visible(wk1gt_4,351). happensAt(inactive(wk1gt_4),351).
orientation(wk1gt_1,149,352). coords(wk1gt_1,138,132,352).  visible(wk1gt_1,352). happensAt(walking(wk1gt_1),352). orientation(wk1gt_4,162,352). coords(wk1gt_4,52,176,352).  visible(wk1gt_4,352). happensAt(inactive(wk1gt_4),352).
orientation(wk1gt_1,149,353). coords(wk1gt_1,138,131,353).  visible(wk1gt_1,353). happensAt(walking(wk1gt_1),353). orientation(wk1gt_4,159,353). coords(wk1gt_4,52,176,353).  visible(wk1gt_4,353). happensAt(inactive(wk1gt_4),353).
orientation(wk1gt_1,149,354). coords(wk1gt_1,138,131,354).  visible(wk1gt_1,354). happensAt(walking(wk1gt_1),354). orientation(wk1gt_4,162,354). coords(wk1gt_4,52,176,354).  visible(wk1gt_4,354). happensAt(inactive(wk1gt_4),354).
orientation(wk1gt_1,149,355). coords(wk1gt_1,136,130,355).  visible(wk1gt_1,355). happensAt(walking(wk1gt_1),355). orientation(wk1gt_4,165,355). coords(wk1gt_4,52,176,355).  visible(wk1gt_4,355). happensAt(inactive(wk1gt_4),355).
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
#show holdsAt/2.