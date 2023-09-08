
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

	orientation(mwt2gt_1,143,337). coords(mwt2gt_1,137,76,337).  visible(mwt2gt_1,337). happensAt(active(mwt2gt_1),337). orientation(mwt2gt_2,145,337). coords(mwt2gt_2,142,87,337).  visible(mwt2gt_2,337). happensAt(active(mwt2gt_2),337).
orientation(mwt2gt_1,143,338). coords(mwt2gt_1,137,76,338).  visible(mwt2gt_1,338). happensAt(active(mwt2gt_1),338). orientation(mwt2gt_2,145,338). coords(mwt2gt_2,142,87,338).  visible(mwt2gt_2,338). happensAt(active(mwt2gt_2),338).
orientation(mwt2gt_1,143,339). coords(mwt2gt_1,137,76,339).  visible(mwt2gt_1,339). happensAt(active(mwt2gt_1),339). orientation(mwt2gt_2,145,339). coords(mwt2gt_2,142,87,339).  visible(mwt2gt_2,339). happensAt(inactive(mwt2gt_2),339).
orientation(mwt2gt_1,143,340). coords(mwt2gt_1,134,77,340).  visible(mwt2gt_1,340). happensAt(active(mwt2gt_1),340). orientation(mwt2gt_2,141,340). coords(mwt2gt_2,142,86,340).  visible(mwt2gt_2,340). happensAt(active(mwt2gt_2),340).
orientation(mwt2gt_1,143,341). coords(mwt2gt_1,134,77,341).  visible(mwt2gt_1,341). happensAt(active(mwt2gt_1),341). orientation(mwt2gt_2,141,341). coords(mwt2gt_2,142,86,341).  visible(mwt2gt_2,341). happensAt(active(mwt2gt_2),341).
orientation(mwt2gt_1,143,342). coords(mwt2gt_1,134,77,342).  visible(mwt2gt_1,342). happensAt(active(mwt2gt_1),342). orientation(mwt2gt_2,141,342). coords(mwt2gt_2,142,86,342).  visible(mwt2gt_2,342). happensAt(active(mwt2gt_2),342).
orientation(mwt2gt_1,143,343). coords(mwt2gt_1,134,77,343).  visible(mwt2gt_1,343). happensAt(active(mwt2gt_1),343). orientation(mwt2gt_2,141,343). coords(mwt2gt_2,142,86,343).  visible(mwt2gt_2,343). happensAt(active(mwt2gt_2),343).
orientation(mwt2gt_1,143,344). coords(mwt2gt_1,134,77,344).  visible(mwt2gt_1,344). happensAt(active(mwt2gt_1),344). orientation(mwt2gt_2,141,344). coords(mwt2gt_2,142,86,344).  visible(mwt2gt_2,344). happensAt(active(mwt2gt_2),344).
orientation(mwt2gt_1,143,345). coords(mwt2gt_1,134,77,345).  visible(mwt2gt_1,345). happensAt(active(mwt2gt_1),345). orientation(mwt2gt_2,141,345). coords(mwt2gt_2,142,86,345).  visible(mwt2gt_2,345). happensAt(active(mwt2gt_2),345).
orientation(mwt2gt_1,143,346). coords(mwt2gt_1,134,77,346).  visible(mwt2gt_1,346). happensAt(active(mwt2gt_1),346). orientation(mwt2gt_2,141,346). coords(mwt2gt_2,142,86,346).  visible(mwt2gt_2,346). happensAt(active(mwt2gt_2),346).
orientation(mwt2gt_1,143,347). coords(mwt2gt_1,134,77,347).  visible(mwt2gt_1,347). happensAt(active(mwt2gt_1),347). orientation(mwt2gt_2,141,347). coords(mwt2gt_2,142,86,347).  visible(mwt2gt_2,347). happensAt(active(mwt2gt_2),347).
orientation(mwt2gt_1,143,348). coords(mwt2gt_1,134,77,348).  visible(mwt2gt_1,348). happensAt(active(mwt2gt_1),348). orientation(mwt2gt_2,141,348). coords(mwt2gt_2,142,86,348).  visible(mwt2gt_2,348). happensAt(active(mwt2gt_2),348).
orientation(mwt2gt_1,143,349). coords(mwt2gt_1,134,77,349).  visible(mwt2gt_1,349). happensAt(active(mwt2gt_1),349). orientation(mwt2gt_2,141,349). coords(mwt2gt_2,142,86,349).  visible(mwt2gt_2,349). happensAt(active(mwt2gt_2),349).
orientation(mwt2gt_1,143,350). coords(mwt2gt_1,134,77,350).  visible(mwt2gt_1,350). happensAt(active(mwt2gt_1),350). orientation(mwt2gt_2,141,350). coords(mwt2gt_2,142,86,350).  visible(mwt2gt_2,350). happensAt(active(mwt2gt_2),350).
orientation(mwt2gt_1,143,351). coords(mwt2gt_1,134,77,351).  visible(mwt2gt_1,351). happensAt(active(mwt2gt_1),351). orientation(mwt2gt_2,141,351). coords(mwt2gt_2,142,86,351).  visible(mwt2gt_2,351). happensAt(active(mwt2gt_2),351).
orientation(mwt2gt_1,143,352). coords(mwt2gt_1,134,77,352).  visible(mwt2gt_1,352). happensAt(active(mwt2gt_1),352). orientation(mwt2gt_2,141,352). coords(mwt2gt_2,142,86,352).  visible(mwt2gt_2,352). happensAt(active(mwt2gt_2),352).
orientation(mwt2gt_1,143,353). coords(mwt2gt_1,134,77,353).  visible(mwt2gt_1,353). happensAt(active(mwt2gt_1),353). orientation(mwt2gt_2,141,353). coords(mwt2gt_2,142,86,353).  visible(mwt2gt_2,353). happensAt(active(mwt2gt_2),353).
orientation(mwt2gt_1,143,354). coords(mwt2gt_1,134,77,354).  visible(mwt2gt_1,354). happensAt(active(mwt2gt_1),354). orientation(mwt2gt_2,141,354). coords(mwt2gt_2,142,86,354).  visible(mwt2gt_2,354). happensAt(active(mwt2gt_2),354).
orientation(mwt2gt_1,143,355). coords(mwt2gt_1,134,77,355).  visible(mwt2gt_1,355). happensAt(active(mwt2gt_1),355). orientation(mwt2gt_2,141,355). coords(mwt2gt_2,142,86,355).  visible(mwt2gt_2,355). happensAt(active(mwt2gt_2),355).
orientation(mwt2gt_1,143,356). coords(mwt2gt_1,134,77,356).  visible(mwt2gt_1,356). happensAt(active(mwt2gt_1),356). orientation(mwt2gt_2,141,356). coords(mwt2gt_2,142,86,356).  visible(mwt2gt_2,356). happensAt(active(mwt2gt_2),356).
orientation(mwt2gt_1,143,357). coords(mwt2gt_1,134,77,357).  visible(mwt2gt_1,357). happensAt(active(mwt2gt_1),357). orientation(mwt2gt_2,141,357). coords(mwt2gt_2,142,86,357).  visible(mwt2gt_2,357). happensAt(active(mwt2gt_2),357).
orientation(mwt2gt_1,143,358). coords(mwt2gt_1,133,77,358).  visible(mwt2gt_1,358). happensAt(active(mwt2gt_1),358). orientation(mwt2gt_2,141,358). coords(mwt2gt_2,141,86,358).  visible(mwt2gt_2,358). happensAt(active(mwt2gt_2),358).
orientation(mwt2gt_1,143,359). coords(mwt2gt_1,132,77,359).  visible(mwt2gt_1,359). happensAt(active(mwt2gt_1),359). orientation(mwt2gt_2,141,359). coords(mwt2gt_2,141,86,359).  visible(mwt2gt_2,359). happensAt(active(mwt2gt_2),359).
orientation(mwt2gt_1,143,360). coords(mwt2gt_1,132,77,360).  visible(mwt2gt_1,360). happensAt(walking(mwt2gt_1),360). orientation(mwt2gt_2,141,360). coords(mwt2gt_2,141,87,360).  visible(mwt2gt_2,360). happensAt(active(mwt2gt_2),360).
orientation(mwt2gt_1,143,361). coords(mwt2gt_1,131,77,361).  visible(mwt2gt_1,361). happensAt(walking(mwt2gt_1),361). orientation(mwt2gt_2,143,361). coords(mwt2gt_2,141,88,361).  visible(mwt2gt_2,361). happensAt(walking(mwt2gt_2),361).
#show holdsAt/2.