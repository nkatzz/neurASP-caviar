
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

	orientation(fcgt_0,152,132). coords(fcgt_0,277,268,132).  visible(fcgt_0,132). happensAt(walking(fcgt_0),132). orientation(fcgt_4,165,132). coords(fcgt_4,101,210,132).  visible(fcgt_4,132). happensAt(inactive(fcgt_4),132).
orientation(fcgt_0,152,133). coords(fcgt_0,280,268,133).  visible(fcgt_0,133). happensAt(walking(fcgt_0),133). orientation(fcgt_4,165,133). coords(fcgt_4,101,210,133).  visible(fcgt_4,133). happensAt(inactive(fcgt_4),133).
orientation(fcgt_0,152,134). coords(fcgt_0,283,268,134).  visible(fcgt_0,134). happensAt(walking(fcgt_0),134). orientation(fcgt_4,170,134). coords(fcgt_4,101,210,134).  visible(fcgt_4,134). happensAt(inactive(fcgt_4),134).
orientation(fcgt_0,152,135). coords(fcgt_0,285,268,135).  visible(fcgt_0,135). happensAt(walking(fcgt_0),135). orientation(fcgt_4,170,135). coords(fcgt_4,101,210,135).  visible(fcgt_4,135). happensAt(inactive(fcgt_4),135).
orientation(fcgt_0,152,136). coords(fcgt_0,287,268,136).  visible(fcgt_0,136). happensAt(walking(fcgt_0),136). orientation(fcgt_4,170,136). coords(fcgt_4,101,210,136).  visible(fcgt_4,136). happensAt(inactive(fcgt_4),136).
orientation(fcgt_0,152,137). coords(fcgt_0,288,268,137).  visible(fcgt_0,137). happensAt(walking(fcgt_0),137). orientation(fcgt_4,170,137). coords(fcgt_4,101,210,137).  visible(fcgt_4,137). happensAt(inactive(fcgt_4),137).
orientation(fcgt_0,152,138). coords(fcgt_0,289,269,138).  visible(fcgt_0,138). happensAt(walking(fcgt_0),138). orientation(fcgt_4,170,138). coords(fcgt_4,101,210,138).  visible(fcgt_4,138). happensAt(inactive(fcgt_4),138).
orientation(fcgt_0,152,139). coords(fcgt_0,290,271,139).  visible(fcgt_0,139). happensAt(walking(fcgt_0),139). orientation(fcgt_4,170,139). coords(fcgt_4,101,210,139).  visible(fcgt_4,139). happensAt(inactive(fcgt_4),139).
orientation(fcgt_0,152,140). coords(fcgt_0,291,273,140).  visible(fcgt_0,140). happensAt(walking(fcgt_0),140). orientation(fcgt_4,170,140). coords(fcgt_4,101,210,140).  visible(fcgt_4,140). happensAt(inactive(fcgt_4),140).
orientation(fcgt_0,152,141). coords(fcgt_0,292,274,141).  visible(fcgt_0,141). happensAt(walking(fcgt_0),141). orientation(fcgt_4,170,141). coords(fcgt_4,101,210,141).  visible(fcgt_4,141). happensAt(inactive(fcgt_4),141).
orientation(fcgt_0,152,142). coords(fcgt_0,293,276,142).  visible(fcgt_0,142). happensAt(walking(fcgt_0),142). orientation(fcgt_4,170,142). coords(fcgt_4,101,210,142).  visible(fcgt_4,142). happensAt(inactive(fcgt_4),142).
orientation(fcgt_0,152,143). coords(fcgt_0,294,278,143).  visible(fcgt_0,143). happensAt(walking(fcgt_0),143). orientation(fcgt_4,170,143). coords(fcgt_4,101,210,143).  visible(fcgt_4,143). happensAt(inactive(fcgt_4),143).
orientation(fcgt_0,152,144). coords(fcgt_0,297,278,144).  visible(fcgt_0,144). happensAt(walking(fcgt_0),144). orientation(fcgt_4,170,144). coords(fcgt_4,101,210,144).  visible(fcgt_4,144). happensAt(inactive(fcgt_4),144).
orientation(fcgt_0,152,145). coords(fcgt_0,298,278,145).  visible(fcgt_0,145). happensAt(walking(fcgt_0),145). orientation(fcgt_4,170,145). coords(fcgt_4,101,210,145).  visible(fcgt_4,145). happensAt(inactive(fcgt_4),145).
orientation(fcgt_0,152,146). coords(fcgt_0,299,278,146).  visible(fcgt_0,146). happensAt(walking(fcgt_0),146). orientation(fcgt_4,170,146). coords(fcgt_4,101,210,146).  visible(fcgt_4,146). happensAt(inactive(fcgt_4),146).
orientation(fcgt_0,152,147). coords(fcgt_0,302,278,147).  visible(fcgt_0,147). happensAt(walking(fcgt_0),147). orientation(fcgt_4,170,147). coords(fcgt_4,101,210,147).  visible(fcgt_4,147). happensAt(inactive(fcgt_4),147).
orientation(fcgt_0,152,148). coords(fcgt_0,305,278,148).  visible(fcgt_0,148). happensAt(walking(fcgt_0),148). orientation(fcgt_4,170,148). coords(fcgt_4,101,210,148).  visible(fcgt_4,148). happensAt(inactive(fcgt_4),148).
orientation(fcgt_0,152,149). coords(fcgt_0,305,278,149).  visible(fcgt_0,149). happensAt(walking(fcgt_0),149). orientation(fcgt_4,170,149). coords(fcgt_4,101,210,149).  visible(fcgt_4,149). happensAt(inactive(fcgt_4),149).
orientation(fcgt_0,152,150). coords(fcgt_0,305,278,150).  visible(fcgt_0,150). happensAt(walking(fcgt_0),150). orientation(fcgt_4,170,150). coords(fcgt_4,101,210,150).  visible(fcgt_4,150). happensAt(walking(fcgt_4),150).
orientation(fcgt_0,152,151). coords(fcgt_0,305,278,151).  visible(fcgt_0,151). happensAt(walking(fcgt_0),151). orientation(fcgt_4,170,151). coords(fcgt_4,101,210,151).  visible(fcgt_4,151). happensAt(walking(fcgt_4),151).
orientation(fcgt_0,152,152). coords(fcgt_0,306,278,152).  visible(fcgt_0,152). happensAt(walking(fcgt_0),152). orientation(fcgt_4,170,152). coords(fcgt_4,101,210,152).  visible(fcgt_4,152). happensAt(walking(fcgt_4),152).
orientation(fcgt_0,152,153). coords(fcgt_0,307,280,153).  visible(fcgt_0,153). happensAt(walking(fcgt_0),153). orientation(fcgt_4,170,153). coords(fcgt_4,101,210,153).  visible(fcgt_4,153). happensAt(walking(fcgt_4),153).
orientation(fcgt_0,152,154). coords(fcgt_0,308,282,154).  visible(fcgt_0,154). happensAt(walking(fcgt_0),154). orientation(fcgt_4,170,154). coords(fcgt_4,101,210,154).  visible(fcgt_4,154). happensAt(walking(fcgt_4),154).
orientation(fcgt_0,152,155). coords(fcgt_0,308,284,155).  visible(fcgt_0,155). happensAt(walking(fcgt_0),155). orientation(fcgt_4,170,155). coords(fcgt_4,100,210,155).  visible(fcgt_4,155). happensAt(walking(fcgt_4),155).
orientation(fcgt_0,152,156). coords(fcgt_0,310,285,156).  happensAt(disappear(fcgt_0),156). happensAt(walking(fcgt_0),156). orientation(fcgt_4,170,156). coords(fcgt_4,100,210,156).  visible(fcgt_4,156). happensAt(walking(fcgt_4),156).
#show holdsAt/2.