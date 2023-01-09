
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

	orientation(rffgt_0,135,132). coords(rffgt_0,153,34,132).  visible(rffgt_0,132). happensAt(walking(rffgt_0),132). orientation(rffgt_1,148,132). coords(rffgt_1,152,109,132).  visible(rffgt_1,132). happensAt(inactive(rffgt_1),132).
orientation(rffgt_0,135,133). coords(rffgt_0,153,34,133).  visible(rffgt_0,133). happensAt(walking(rffgt_0),133). orientation(rffgt_1,148,133). coords(rffgt_1,152,109,133).  visible(rffgt_1,133). happensAt(inactive(rffgt_1),133).
orientation(rffgt_0,135,134). coords(rffgt_0,154,34,134).  visible(rffgt_0,134). happensAt(walking(rffgt_0),134). orientation(rffgt_1,148,134). coords(rffgt_1,152,109,134).  visible(rffgt_1,134). happensAt(inactive(rffgt_1),134).
orientation(rffgt_0,135,135). coords(rffgt_0,155,34,135).  visible(rffgt_0,135). happensAt(walking(rffgt_0),135). orientation(rffgt_1,148,135). coords(rffgt_1,152,109,135).  visible(rffgt_1,135). happensAt(inactive(rffgt_1),135).
orientation(rffgt_0,135,136). coords(rffgt_0,155,34,136).  visible(rffgt_0,136). happensAt(walking(rffgt_0),136). orientation(rffgt_1,148,136). coords(rffgt_1,152,109,136).  visible(rffgt_1,136). happensAt(inactive(rffgt_1),136).
orientation(rffgt_0,135,137). coords(rffgt_0,157,34,137).  visible(rffgt_0,137). happensAt(walking(rffgt_0),137). orientation(rffgt_1,148,137). coords(rffgt_1,152,109,137).  visible(rffgt_1,137). happensAt(inactive(rffgt_1),137).
orientation(rffgt_0,135,138). coords(rffgt_0,157,34,138).  visible(rffgt_0,138). happensAt(walking(rffgt_0),138). orientation(rffgt_1,148,138). coords(rffgt_1,152,109,138).  visible(rffgt_1,138). happensAt(inactive(rffgt_1),138).
orientation(rffgt_0,135,139). coords(rffgt_0,157,33,139).  visible(rffgt_0,139). happensAt(walking(rffgt_0),139). orientation(rffgt_1,148,139). coords(rffgt_1,152,109,139).  visible(rffgt_1,139). happensAt(inactive(rffgt_1),139).
orientation(rffgt_0,135,140). coords(rffgt_0,157,33,140).  visible(rffgt_0,140). happensAt(walking(rffgt_0),140). orientation(rffgt_1,148,140). coords(rffgt_1,152,109,140).  visible(rffgt_1,140). happensAt(inactive(rffgt_1),140).
orientation(rffgt_0,135,141). coords(rffgt_0,157,32,141).  visible(rffgt_0,141). happensAt(walking(rffgt_0),141). orientation(rffgt_1,148,141). coords(rffgt_1,152,109,141).  visible(rffgt_1,141). happensAt(inactive(rffgt_1),141).
orientation(rffgt_0,135,142). coords(rffgt_0,158,32,142).  visible(rffgt_0,142). happensAt(walking(rffgt_0),142). orientation(rffgt_1,148,142). coords(rffgt_1,152,109,142).  visible(rffgt_1,142). happensAt(inactive(rffgt_1),142).
orientation(rffgt_0,135,143). coords(rffgt_0,158,32,143).  visible(rffgt_0,143). happensAt(walking(rffgt_0),143). orientation(rffgt_1,148,143). coords(rffgt_1,152,109,143).  visible(rffgt_1,143). happensAt(inactive(rffgt_1),143).
orientation(rffgt_0,135,144). coords(rffgt_0,158,32,144).  visible(rffgt_0,144). happensAt(walking(rffgt_0),144). orientation(rffgt_1,148,144). coords(rffgt_1,152,109,144).  visible(rffgt_1,144). happensAt(inactive(rffgt_1),144).
orientation(rffgt_0,135,145). coords(rffgt_0,159,31,145).  visible(rffgt_0,145). happensAt(walking(rffgt_0),145). orientation(rffgt_1,148,145). coords(rffgt_1,152,109,145).  visible(rffgt_1,145). happensAt(inactive(rffgt_1),145).
orientation(rffgt_0,135,146). coords(rffgt_0,158,30,146).  visible(rffgt_0,146). happensAt(walking(rffgt_0),146). orientation(rffgt_1,148,146). coords(rffgt_1,152,109,146).  visible(rffgt_1,146). happensAt(inactive(rffgt_1),146).
orientation(rffgt_0,135,147). coords(rffgt_0,159,30,147).  visible(rffgt_0,147). happensAt(walking(rffgt_0),147). orientation(rffgt_1,148,147). coords(rffgt_1,152,109,147).  visible(rffgt_1,147). happensAt(inactive(rffgt_1),147).
orientation(rffgt_0,135,148). coords(rffgt_0,160,30,148).  visible(rffgt_0,148). happensAt(walking(rffgt_0),148). orientation(rffgt_1,148,148). coords(rffgt_1,152,109,148).  visible(rffgt_1,148). happensAt(inactive(rffgt_1),148).
orientation(rffgt_0,135,149). coords(rffgt_0,162,30,149).  visible(rffgt_0,149). happensAt(walking(rffgt_0),149). orientation(rffgt_1,148,149). coords(rffgt_1,152,109,149).  visible(rffgt_1,149). happensAt(inactive(rffgt_1),149).
orientation(rffgt_0,135,150). coords(rffgt_0,163,29,150).  visible(rffgt_0,150). happensAt(walking(rffgt_0),150). orientation(rffgt_1,148,150). coords(rffgt_1,152,109,150).  visible(rffgt_1,150). happensAt(inactive(rffgt_1),150).
orientation(rffgt_0,135,151). coords(rffgt_0,163,29,151).  visible(rffgt_0,151). happensAt(walking(rffgt_0),151). orientation(rffgt_1,148,151). coords(rffgt_1,152,109,151).  visible(rffgt_1,151). happensAt(inactive(rffgt_1),151).
orientation(rffgt_0,135,152). coords(rffgt_0,164,29,152).  visible(rffgt_0,152). happensAt(walking(rffgt_0),152). orientation(rffgt_1,148,152). coords(rffgt_1,152,109,152).  visible(rffgt_1,152). happensAt(inactive(rffgt_1),152).
orientation(rffgt_0,135,153). coords(rffgt_0,164,29,153).  visible(rffgt_0,153). happensAt(walking(rffgt_0),153). orientation(rffgt_1,148,153). coords(rffgt_1,152,109,153).  visible(rffgt_1,153). happensAt(inactive(rffgt_1),153).
orientation(rffgt_0,135,154). coords(rffgt_0,164,28,154).  visible(rffgt_0,154). happensAt(walking(rffgt_0),154). orientation(rffgt_1,148,154). coords(rffgt_1,152,109,154).  visible(rffgt_1,154). happensAt(inactive(rffgt_1),154).
orientation(rffgt_0,135,155). coords(rffgt_0,164,28,155).  visible(rffgt_0,155). happensAt(walking(rffgt_0),155). orientation(rffgt_1,148,155). coords(rffgt_1,152,109,155).  visible(rffgt_1,155). happensAt(inactive(rffgt_1),155).
orientation(rffgt_0,131,156). coords(rffgt_0,165,28,156).  visible(rffgt_0,156). happensAt(walking(rffgt_0),156). orientation(rffgt_1,148,156). coords(rffgt_1,152,109,156).  visible(rffgt_1,156). happensAt(inactive(rffgt_1),156).
#show holdsAt/2.