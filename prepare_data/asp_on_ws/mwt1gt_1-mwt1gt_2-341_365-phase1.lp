
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

	orientation(mwt1gt_1,160,341). coords(mwt1gt_1,118,148,341).  visible(mwt1gt_1,341). happensAt(active(mwt1gt_1),341). orientation(mwt1gt_2,140,341). coords(mwt1gt_2,127,127,341).  visible(mwt1gt_2,341). happensAt(active(mwt1gt_2),341).
orientation(mwt1gt_1,160,342). coords(mwt1gt_1,118,148,342).  visible(mwt1gt_1,342). happensAt(active(mwt1gt_1),342). orientation(mwt1gt_2,140,342). coords(mwt1gt_2,127,127,342).  visible(mwt1gt_2,342). happensAt(active(mwt1gt_2),342).
orientation(mwt1gt_1,160,343). coords(mwt1gt_1,118,148,343).  visible(mwt1gt_1,343). happensAt(active(mwt1gt_1),343). orientation(mwt1gt_2,140,343). coords(mwt1gt_2,126,127,343).  visible(mwt1gt_2,343). happensAt(active(mwt1gt_2),343).
orientation(mwt1gt_1,160,344). coords(mwt1gt_1,118,148,344).  visible(mwt1gt_1,344). happensAt(active(mwt1gt_1),344). orientation(mwt1gt_2,140,344). coords(mwt1gt_2,126,127,344).  visible(mwt1gt_2,344). happensAt(active(mwt1gt_2),344).
orientation(mwt1gt_1,160,345). coords(mwt1gt_1,118,148,345).  visible(mwt1gt_1,345). happensAt(active(mwt1gt_1),345). orientation(mwt1gt_2,140,345). coords(mwt1gt_2,125,127,345).  visible(mwt1gt_2,345). happensAt(active(mwt1gt_2),345).
orientation(mwt1gt_1,160,346). coords(mwt1gt_1,117,148,346).  visible(mwt1gt_1,346). happensAt(active(mwt1gt_1),346). orientation(mwt1gt_2,140,346). coords(mwt1gt_2,125,127,346).  visible(mwt1gt_2,346). happensAt(active(mwt1gt_2),346).
orientation(mwt1gt_1,160,347). coords(mwt1gt_1,117,148,347).  visible(mwt1gt_1,347). happensAt(active(mwt1gt_1),347). orientation(mwt1gt_2,140,347). coords(mwt1gt_2,124,127,347).  visible(mwt1gt_2,347). happensAt(active(mwt1gt_2),347).
orientation(mwt1gt_1,160,348). coords(mwt1gt_1,117,148,348).  visible(mwt1gt_1,348). happensAt(active(mwt1gt_1),348). orientation(mwt1gt_2,140,348). coords(mwt1gt_2,124,127,348).  visible(mwt1gt_2,348). happensAt(active(mwt1gt_2),348).
orientation(mwt1gt_1,160,349). coords(mwt1gt_1,116,148,349).  visible(mwt1gt_1,349). happensAt(active(mwt1gt_1),349). orientation(mwt1gt_2,140,349). coords(mwt1gt_2,124,126,349).  visible(mwt1gt_2,349). happensAt(active(mwt1gt_2),349).
orientation(mwt1gt_1,160,350). coords(mwt1gt_1,116,148,350).  visible(mwt1gt_1,350). happensAt(active(mwt1gt_1),350). orientation(mwt1gt_2,140,350). coords(mwt1gt_2,124,126,350).  visible(mwt1gt_2,350). happensAt(active(mwt1gt_2),350).
orientation(mwt1gt_1,160,351). coords(mwt1gt_1,115,147,351).  visible(mwt1gt_1,351). happensAt(active(mwt1gt_1),351). orientation(mwt1gt_2,140,351). coords(mwt1gt_2,123,126,351).  visible(mwt1gt_2,351). happensAt(active(mwt1gt_2),351).
orientation(mwt1gt_1,160,352). coords(mwt1gt_1,115,147,352).  visible(mwt1gt_1,352). happensAt(active(mwt1gt_1),352). orientation(mwt1gt_2,140,352). coords(mwt1gt_2,121,125,352).  visible(mwt1gt_2,352). happensAt(active(mwt1gt_2),352).
orientation(mwt1gt_1,160,353). coords(mwt1gt_1,115,147,353).  visible(mwt1gt_1,353). happensAt(active(mwt1gt_1),353). orientation(mwt1gt_2,140,353). coords(mwt1gt_2,121,125,353).  visible(mwt1gt_2,353). happensAt(active(mwt1gt_2),353).
orientation(mwt1gt_1,160,354). coords(mwt1gt_1,115,147,354).  visible(mwt1gt_1,354). happensAt(active(mwt1gt_1),354). orientation(mwt1gt_2,140,354). coords(mwt1gt_2,121,125,354).  visible(mwt1gt_2,354). happensAt(walking(mwt1gt_2),354).
orientation(mwt1gt_1,160,355). coords(mwt1gt_1,111,147,355).  visible(mwt1gt_1,355). happensAt(active(mwt1gt_1),355). orientation(mwt1gt_2,140,355). coords(mwt1gt_2,118,124,355).  visible(mwt1gt_2,355). happensAt(walking(mwt1gt_2),355).
orientation(mwt1gt_1,160,356). coords(mwt1gt_1,111,147,356).  visible(mwt1gt_1,356). happensAt(walking(mwt1gt_1),356). orientation(mwt1gt_2,140,356). coords(mwt1gt_2,118,124,356).  visible(mwt1gt_2,356). happensAt(walking(mwt1gt_2),356).
orientation(mwt1gt_1,160,357). coords(mwt1gt_1,111,147,357).  visible(mwt1gt_1,357). happensAt(walking(mwt1gt_1),357). orientation(mwt1gt_2,140,357). coords(mwt1gt_2,117,124,357).  visible(mwt1gt_2,357). happensAt(walking(mwt1gt_2),357).
orientation(mwt1gt_1,160,358). coords(mwt1gt_1,111,146,358).  visible(mwt1gt_1,358). happensAt(walking(mwt1gt_1),358). orientation(mwt1gt_2,140,358). coords(mwt1gt_2,117,124,358).  visible(mwt1gt_2,358). happensAt(walking(mwt1gt_2),358).
orientation(mwt1gt_1,160,359). coords(mwt1gt_1,111,146,359).  visible(mwt1gt_1,359). happensAt(walking(mwt1gt_1),359). orientation(mwt1gt_2,140,359). coords(mwt1gt_2,117,124,359).  visible(mwt1gt_2,359). happensAt(walking(mwt1gt_2),359).
orientation(mwt1gt_1,160,360). coords(mwt1gt_1,111,146,360).  visible(mwt1gt_1,360). happensAt(walking(mwt1gt_1),360). orientation(mwt1gt_2,140,360). coords(mwt1gt_2,117,124,360).  visible(mwt1gt_2,360). happensAt(walking(mwt1gt_2),360).
orientation(mwt1gt_1,160,361). coords(mwt1gt_1,111,146,361).  visible(mwt1gt_1,361). happensAt(walking(mwt1gt_1),361). orientation(mwt1gt_2,140,361). coords(mwt1gt_2,117,124,361).  visible(mwt1gt_2,361). happensAt(walking(mwt1gt_2),361).
orientation(mwt1gt_1,160,362). coords(mwt1gt_1,111,145,362).  visible(mwt1gt_1,362). happensAt(walking(mwt1gt_1),362). orientation(mwt1gt_2,140,362). coords(mwt1gt_2,117,124,362).  visible(mwt1gt_2,362). happensAt(walking(mwt1gt_2),362).
orientation(mwt1gt_1,160,363). coords(mwt1gt_1,111,145,363).  visible(mwt1gt_1,363). happensAt(walking(mwt1gt_1),363). orientation(mwt1gt_2,140,363). coords(mwt1gt_2,116,124,363).  visible(mwt1gt_2,363). happensAt(walking(mwt1gt_2),363).
orientation(mwt1gt_1,160,364). coords(mwt1gt_1,109,144,364).  visible(mwt1gt_1,364). happensAt(walking(mwt1gt_1),364). orientation(mwt1gt_2,140,364). coords(mwt1gt_2,116,124,364).  visible(mwt1gt_2,364). happensAt(walking(mwt1gt_2),364).
orientation(mwt1gt_1,160,365). coords(mwt1gt_1,109,144,365).  visible(mwt1gt_1,365). happensAt(walking(mwt1gt_1),365). orientation(mwt1gt_2,140,365). coords(mwt1gt_2,116,124,365).  visible(mwt1gt_2,365). happensAt(walking(mwt1gt_2),365).
#show holdsAt/2.