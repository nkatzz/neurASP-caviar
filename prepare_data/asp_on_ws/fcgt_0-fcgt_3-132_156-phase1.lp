
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

	orientation(fcgt_0,152,132). coords(fcgt_0,277,268,132).  visible(fcgt_0,132). happensAt(walking(fcgt_0),132). orientation(fcgt_3,159,132). coords(fcgt_3,103,163,132).  visible(fcgt_3,132). happensAt(inactive(fcgt_3),132).
orientation(fcgt_0,152,133). coords(fcgt_0,280,268,133).  visible(fcgt_0,133). happensAt(walking(fcgt_0),133). orientation(fcgt_3,159,133). coords(fcgt_3,103,163,133).  visible(fcgt_3,133). happensAt(inactive(fcgt_3),133).
orientation(fcgt_0,152,134). coords(fcgt_0,283,268,134).  visible(fcgt_0,134). happensAt(walking(fcgt_0),134). orientation(fcgt_3,159,134). coords(fcgt_3,103,163,134).  visible(fcgt_3,134). happensAt(inactive(fcgt_3),134).
orientation(fcgt_0,152,135). coords(fcgt_0,285,268,135).  visible(fcgt_0,135). happensAt(walking(fcgt_0),135). orientation(fcgt_3,159,135). coords(fcgt_3,103,163,135).  visible(fcgt_3,135). happensAt(inactive(fcgt_3),135).
orientation(fcgt_0,152,136). coords(fcgt_0,287,268,136).  visible(fcgt_0,136). happensAt(walking(fcgt_0),136). orientation(fcgt_3,159,136). coords(fcgt_3,103,163,136).  visible(fcgt_3,136). happensAt(inactive(fcgt_3),136).
orientation(fcgt_0,152,137). coords(fcgt_0,288,268,137).  visible(fcgt_0,137). happensAt(walking(fcgt_0),137). orientation(fcgt_3,159,137). coords(fcgt_3,103,163,137).  visible(fcgt_3,137). happensAt(inactive(fcgt_3),137).
orientation(fcgt_0,152,138). coords(fcgt_0,289,269,138).  visible(fcgt_0,138). happensAt(walking(fcgt_0),138). orientation(fcgt_3,159,138). coords(fcgt_3,103,163,138).  visible(fcgt_3,138). happensAt(inactive(fcgt_3),138).
orientation(fcgt_0,152,139). coords(fcgt_0,290,271,139).  visible(fcgt_0,139). happensAt(walking(fcgt_0),139). orientation(fcgt_3,159,139). coords(fcgt_3,103,163,139).  visible(fcgt_3,139). happensAt(inactive(fcgt_3),139).
orientation(fcgt_0,152,140). coords(fcgt_0,291,273,140).  visible(fcgt_0,140). happensAt(walking(fcgt_0),140). orientation(fcgt_3,159,140). coords(fcgt_3,103,163,140).  visible(fcgt_3,140). happensAt(inactive(fcgt_3),140).
orientation(fcgt_0,152,141). coords(fcgt_0,292,274,141).  visible(fcgt_0,141). happensAt(walking(fcgt_0),141). orientation(fcgt_3,159,141). coords(fcgt_3,103,163,141).  visible(fcgt_3,141). happensAt(inactive(fcgt_3),141).
orientation(fcgt_0,152,142). coords(fcgt_0,293,276,142).  visible(fcgt_0,142). happensAt(walking(fcgt_0),142). orientation(fcgt_3,159,142). coords(fcgt_3,103,163,142).  visible(fcgt_3,142). happensAt(inactive(fcgt_3),142).
orientation(fcgt_0,152,143). coords(fcgt_0,294,278,143).  visible(fcgt_0,143). happensAt(walking(fcgt_0),143). orientation(fcgt_3,159,143). coords(fcgt_3,103,163,143).  visible(fcgt_3,143). happensAt(inactive(fcgt_3),143).
orientation(fcgt_0,152,144). coords(fcgt_0,297,278,144).  visible(fcgt_0,144). happensAt(walking(fcgt_0),144). orientation(fcgt_3,159,144). coords(fcgt_3,103,163,144).  visible(fcgt_3,144). happensAt(inactive(fcgt_3),144).
orientation(fcgt_0,152,145). coords(fcgt_0,298,278,145).  visible(fcgt_0,145). happensAt(walking(fcgt_0),145). orientation(fcgt_3,159,145). coords(fcgt_3,103,163,145).  visible(fcgt_3,145). happensAt(inactive(fcgt_3),145).
orientation(fcgt_0,152,146). coords(fcgt_0,299,278,146).  visible(fcgt_0,146). happensAt(walking(fcgt_0),146). orientation(fcgt_3,159,146). coords(fcgt_3,103,163,146).  visible(fcgt_3,146). happensAt(inactive(fcgt_3),146).
orientation(fcgt_0,152,147). coords(fcgt_0,302,278,147).  visible(fcgt_0,147). happensAt(walking(fcgt_0),147). orientation(fcgt_3,159,147). coords(fcgt_3,103,163,147).  visible(fcgt_3,147). happensAt(inactive(fcgt_3),147).
orientation(fcgt_0,152,148). coords(fcgt_0,305,278,148).  visible(fcgt_0,148). happensAt(walking(fcgt_0),148). orientation(fcgt_3,159,148). coords(fcgt_3,103,163,148).  visible(fcgt_3,148). happensAt(inactive(fcgt_3),148).
orientation(fcgt_0,152,149). coords(fcgt_0,305,278,149).  visible(fcgt_0,149). happensAt(walking(fcgt_0),149). orientation(fcgt_3,159,149). coords(fcgt_3,103,163,149).  visible(fcgt_3,149). happensAt(inactive(fcgt_3),149).
orientation(fcgt_0,152,150). coords(fcgt_0,305,278,150).  visible(fcgt_0,150). happensAt(walking(fcgt_0),150). orientation(fcgt_3,159,150). coords(fcgt_3,103,163,150).  visible(fcgt_3,150). happensAt(inactive(fcgt_3),150).
orientation(fcgt_0,152,151). coords(fcgt_0,305,278,151).  visible(fcgt_0,151). happensAt(walking(fcgt_0),151). orientation(fcgt_3,159,151). coords(fcgt_3,103,163,151).  visible(fcgt_3,151). happensAt(inactive(fcgt_3),151).
orientation(fcgt_0,152,152). coords(fcgt_0,306,278,152).  visible(fcgt_0,152). happensAt(walking(fcgt_0),152). orientation(fcgt_3,159,152). coords(fcgt_3,103,163,152).  visible(fcgt_3,152). happensAt(inactive(fcgt_3),152).
orientation(fcgt_0,152,153). coords(fcgt_0,307,280,153).  visible(fcgt_0,153). happensAt(walking(fcgt_0),153). orientation(fcgt_3,159,153). coords(fcgt_3,103,163,153).  visible(fcgt_3,153). happensAt(inactive(fcgt_3),153).
orientation(fcgt_0,152,154). coords(fcgt_0,308,282,154).  visible(fcgt_0,154). happensAt(walking(fcgt_0),154). orientation(fcgt_3,159,154). coords(fcgt_3,103,163,154).  visible(fcgt_3,154). happensAt(inactive(fcgt_3),154).
orientation(fcgt_0,152,155). coords(fcgt_0,308,284,155).  visible(fcgt_0,155). happensAt(walking(fcgt_0),155). orientation(fcgt_3,159,155). coords(fcgt_3,103,163,155).  visible(fcgt_3,155). happensAt(inactive(fcgt_3),155).
orientation(fcgt_0,152,156). coords(fcgt_0,310,285,156).  happensAt(disappear(fcgt_0),156). happensAt(walking(fcgt_0),156). orientation(fcgt_3,159,156). coords(fcgt_3,103,163,156).  visible(fcgt_3,156). happensAt(inactive(fcgt_3),156).
#show holdsAt/2.