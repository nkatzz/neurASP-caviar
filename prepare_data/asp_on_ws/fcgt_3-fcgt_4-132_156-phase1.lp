
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

	orientation(fcgt_3,159,132). coords(fcgt_3,103,163,132).  visible(fcgt_3,132). happensAt(inactive(fcgt_3),132). orientation(fcgt_4,165,132). coords(fcgt_4,101,210,132).  visible(fcgt_4,132). happensAt(inactive(fcgt_4),132).
orientation(fcgt_3,159,133). coords(fcgt_3,103,163,133).  visible(fcgt_3,133). happensAt(inactive(fcgt_3),133). orientation(fcgt_4,165,133). coords(fcgt_4,101,210,133).  visible(fcgt_4,133). happensAt(inactive(fcgt_4),133).
orientation(fcgt_3,159,134). coords(fcgt_3,103,163,134).  visible(fcgt_3,134). happensAt(inactive(fcgt_3),134). orientation(fcgt_4,170,134). coords(fcgt_4,101,210,134).  visible(fcgt_4,134). happensAt(inactive(fcgt_4),134).
orientation(fcgt_3,159,135). coords(fcgt_3,103,163,135).  visible(fcgt_3,135). happensAt(inactive(fcgt_3),135). orientation(fcgt_4,170,135). coords(fcgt_4,101,210,135).  visible(fcgt_4,135). happensAt(inactive(fcgt_4),135).
orientation(fcgt_3,159,136). coords(fcgt_3,103,163,136).  visible(fcgt_3,136). happensAt(inactive(fcgt_3),136). orientation(fcgt_4,170,136). coords(fcgt_4,101,210,136).  visible(fcgt_4,136). happensAt(inactive(fcgt_4),136).
orientation(fcgt_3,159,137). coords(fcgt_3,103,163,137).  visible(fcgt_3,137). happensAt(inactive(fcgt_3),137). orientation(fcgt_4,170,137). coords(fcgt_4,101,210,137).  visible(fcgt_4,137). happensAt(inactive(fcgt_4),137).
orientation(fcgt_3,159,138). coords(fcgt_3,103,163,138).  visible(fcgt_3,138). happensAt(inactive(fcgt_3),138). orientation(fcgt_4,170,138). coords(fcgt_4,101,210,138).  visible(fcgt_4,138). happensAt(inactive(fcgt_4),138).
orientation(fcgt_3,159,139). coords(fcgt_3,103,163,139).  visible(fcgt_3,139). happensAt(inactive(fcgt_3),139). orientation(fcgt_4,170,139). coords(fcgt_4,101,210,139).  visible(fcgt_4,139). happensAt(inactive(fcgt_4),139).
orientation(fcgt_3,159,140). coords(fcgt_3,103,163,140).  visible(fcgt_3,140). happensAt(inactive(fcgt_3),140). orientation(fcgt_4,170,140). coords(fcgt_4,101,210,140).  visible(fcgt_4,140). happensAt(inactive(fcgt_4),140).
orientation(fcgt_3,159,141). coords(fcgt_3,103,163,141).  visible(fcgt_3,141). happensAt(inactive(fcgt_3),141). orientation(fcgt_4,170,141). coords(fcgt_4,101,210,141).  visible(fcgt_4,141). happensAt(inactive(fcgt_4),141).
orientation(fcgt_3,159,142). coords(fcgt_3,103,163,142).  visible(fcgt_3,142). happensAt(inactive(fcgt_3),142). orientation(fcgt_4,170,142). coords(fcgt_4,101,210,142).  visible(fcgt_4,142). happensAt(inactive(fcgt_4),142).
orientation(fcgt_3,159,143). coords(fcgt_3,103,163,143).  visible(fcgt_3,143). happensAt(inactive(fcgt_3),143). orientation(fcgt_4,170,143). coords(fcgt_4,101,210,143).  visible(fcgt_4,143). happensAt(inactive(fcgt_4),143).
orientation(fcgt_3,159,144). coords(fcgt_3,103,163,144).  visible(fcgt_3,144). happensAt(inactive(fcgt_3),144). orientation(fcgt_4,170,144). coords(fcgt_4,101,210,144).  visible(fcgt_4,144). happensAt(inactive(fcgt_4),144).
orientation(fcgt_3,159,145). coords(fcgt_3,103,163,145).  visible(fcgt_3,145). happensAt(inactive(fcgt_3),145). orientation(fcgt_4,170,145). coords(fcgt_4,101,210,145).  visible(fcgt_4,145). happensAt(inactive(fcgt_4),145).
orientation(fcgt_3,159,146). coords(fcgt_3,103,163,146).  visible(fcgt_3,146). happensAt(inactive(fcgt_3),146). orientation(fcgt_4,170,146). coords(fcgt_4,101,210,146).  visible(fcgt_4,146). happensAt(inactive(fcgt_4),146).
orientation(fcgt_3,159,147). coords(fcgt_3,103,163,147).  visible(fcgt_3,147). happensAt(inactive(fcgt_3),147). orientation(fcgt_4,170,147). coords(fcgt_4,101,210,147).  visible(fcgt_4,147). happensAt(inactive(fcgt_4),147).
orientation(fcgt_3,159,148). coords(fcgt_3,103,163,148).  visible(fcgt_3,148). happensAt(inactive(fcgt_3),148). orientation(fcgt_4,170,148). coords(fcgt_4,101,210,148).  visible(fcgt_4,148). happensAt(inactive(fcgt_4),148).
orientation(fcgt_3,159,149). coords(fcgt_3,103,163,149).  visible(fcgt_3,149). happensAt(inactive(fcgt_3),149). orientation(fcgt_4,170,149). coords(fcgt_4,101,210,149).  visible(fcgt_4,149). happensAt(inactive(fcgt_4),149).
orientation(fcgt_3,159,150). coords(fcgt_3,103,163,150).  visible(fcgt_3,150). happensAt(inactive(fcgt_3),150). orientation(fcgt_4,170,150). coords(fcgt_4,101,210,150).  visible(fcgt_4,150). happensAt(walking(fcgt_4),150).
orientation(fcgt_3,159,151). coords(fcgt_3,103,163,151).  visible(fcgt_3,151). happensAt(inactive(fcgt_3),151). orientation(fcgt_4,170,151). coords(fcgt_4,101,210,151).  visible(fcgt_4,151). happensAt(walking(fcgt_4),151).
orientation(fcgt_3,159,152). coords(fcgt_3,103,163,152).  visible(fcgt_3,152). happensAt(inactive(fcgt_3),152). orientation(fcgt_4,170,152). coords(fcgt_4,101,210,152).  visible(fcgt_4,152). happensAt(walking(fcgt_4),152).
orientation(fcgt_3,159,153). coords(fcgt_3,103,163,153).  visible(fcgt_3,153). happensAt(inactive(fcgt_3),153). orientation(fcgt_4,170,153). coords(fcgt_4,101,210,153).  visible(fcgt_4,153). happensAt(walking(fcgt_4),153).
orientation(fcgt_3,159,154). coords(fcgt_3,103,163,154).  visible(fcgt_3,154). happensAt(inactive(fcgt_3),154). orientation(fcgt_4,170,154). coords(fcgt_4,101,210,154).  visible(fcgt_4,154). happensAt(walking(fcgt_4),154).
orientation(fcgt_3,159,155). coords(fcgt_3,103,163,155).  visible(fcgt_3,155). happensAt(inactive(fcgt_3),155). orientation(fcgt_4,170,155). coords(fcgt_4,100,210,155).  visible(fcgt_4,155). happensAt(walking(fcgt_4),155).
orientation(fcgt_3,159,156). coords(fcgt_3,103,163,156).  visible(fcgt_3,156). happensAt(inactive(fcgt_3),156). orientation(fcgt_4,170,156). coords(fcgt_4,100,210,156).  visible(fcgt_4,156). happensAt(walking(fcgt_4),156).
#show holdsAt/2.