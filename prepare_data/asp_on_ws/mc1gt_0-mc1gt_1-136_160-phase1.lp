
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

	orientation(mc1gt_0,147,136). coords(mc1gt_0,92,106,136).  visible(mc1gt_0,136). happensAt(walking(mc1gt_0),136). orientation(mc1gt_1,145,136). coords(mc1gt_1,113,81,136).  visible(mc1gt_1,136). happensAt(walking(mc1gt_1),136).
orientation(mc1gt_0,147,137). coords(mc1gt_0,93,106,137).  visible(mc1gt_0,137). happensAt(walking(mc1gt_0),137). orientation(mc1gt_1,145,137). coords(mc1gt_1,113,81,137).  visible(mc1gt_1,137). happensAt(walking(mc1gt_1),137).
orientation(mc1gt_0,147,138). coords(mc1gt_0,94,106,138).  visible(mc1gt_0,138). happensAt(walking(mc1gt_0),138). orientation(mc1gt_1,145,138). coords(mc1gt_1,114,82,138).  visible(mc1gt_1,138). happensAt(walking(mc1gt_1),138).
orientation(mc1gt_0,147,139). coords(mc1gt_0,95,106,139).  visible(mc1gt_0,139). happensAt(walking(mc1gt_0),139). orientation(mc1gt_1,145,139). coords(mc1gt_1,116,82,139).  visible(mc1gt_1,139). happensAt(walking(mc1gt_1),139).
orientation(mc1gt_0,147,140). coords(mc1gt_0,95,106,140).  visible(mc1gt_0,140). happensAt(walking(mc1gt_0),140). orientation(mc1gt_1,145,140). coords(mc1gt_1,116,82,140).  visible(mc1gt_1,140). happensAt(walking(mc1gt_1),140).
orientation(mc1gt_0,147,141). coords(mc1gt_0,95,106,141).  visible(mc1gt_0,141). happensAt(walking(mc1gt_0),141). orientation(mc1gt_1,145,141). coords(mc1gt_1,117,82,141).  visible(mc1gt_1,141). happensAt(walking(mc1gt_1),141).
orientation(mc1gt_0,147,142). coords(mc1gt_0,95,106,142).  visible(mc1gt_0,142). happensAt(walking(mc1gt_0),142). orientation(mc1gt_1,145,142). coords(mc1gt_1,118,83,142).  visible(mc1gt_1,142). happensAt(walking(mc1gt_1),142).
orientation(mc1gt_0,147,143). coords(mc1gt_0,95,106,143).  visible(mc1gt_0,143). happensAt(walking(mc1gt_0),143). orientation(mc1gt_1,145,143). coords(mc1gt_1,118,83,143).  visible(mc1gt_1,143). happensAt(walking(mc1gt_1),143).
orientation(mc1gt_0,147,144). coords(mc1gt_0,95,106,144).  visible(mc1gt_0,144). happensAt(walking(mc1gt_0),144). orientation(mc1gt_1,145,144). coords(mc1gt_1,119,83,144).  visible(mc1gt_1,144). happensAt(walking(mc1gt_1),144).
orientation(mc1gt_0,147,145). coords(mc1gt_0,95,107,145).  visible(mc1gt_0,145). happensAt(walking(mc1gt_0),145). orientation(mc1gt_1,145,145). coords(mc1gt_1,119,83,145).  visible(mc1gt_1,145). happensAt(walking(mc1gt_1),145).
orientation(mc1gt_0,147,146). coords(mc1gt_0,96,107,146).  visible(mc1gt_0,146). happensAt(walking(mc1gt_0),146). orientation(mc1gt_1,145,146). coords(mc1gt_1,119,83,146).  visible(mc1gt_1,146). happensAt(walking(mc1gt_1),146).
orientation(mc1gt_0,147,147). coords(mc1gt_0,98,106,147).  visible(mc1gt_0,147). happensAt(walking(mc1gt_0),147). orientation(mc1gt_1,145,147). coords(mc1gt_1,120,83,147).  visible(mc1gt_1,147). happensAt(walking(mc1gt_1),147).
orientation(mc1gt_0,147,148). coords(mc1gt_0,100,107,148).  visible(mc1gt_0,148). happensAt(walking(mc1gt_0),148). orientation(mc1gt_1,145,148). coords(mc1gt_1,120,83,148).  visible(mc1gt_1,148). happensAt(walking(mc1gt_1),148).
orientation(mc1gt_0,147,149). coords(mc1gt_0,101,107,149).  visible(mc1gt_0,149). happensAt(walking(mc1gt_0),149). orientation(mc1gt_1,145,149). coords(mc1gt_1,120,83,149).  visible(mc1gt_1,149). happensAt(walking(mc1gt_1),149).
orientation(mc1gt_0,147,150). coords(mc1gt_0,101,107,150).  visible(mc1gt_0,150). happensAt(walking(mc1gt_0),150). orientation(mc1gt_1,145,150). coords(mc1gt_1,120,83,150).  visible(mc1gt_1,150). happensAt(walking(mc1gt_1),150).
orientation(mc1gt_0,147,151). coords(mc1gt_0,102,107,151).  visible(mc1gt_0,151). happensAt(walking(mc1gt_0),151). orientation(mc1gt_1,145,151). coords(mc1gt_1,121,83,151).  visible(mc1gt_1,151). happensAt(walking(mc1gt_1),151).
orientation(mc1gt_0,147,152). coords(mc1gt_0,103,107,152).  visible(mc1gt_0,152). happensAt(walking(mc1gt_0),152). orientation(mc1gt_1,145,152). coords(mc1gt_1,123,83,152).  visible(mc1gt_1,152). happensAt(walking(mc1gt_1),152).
orientation(mc1gt_0,147,153). coords(mc1gt_0,104,107,153).  visible(mc1gt_0,153). happensAt(walking(mc1gt_0),153). orientation(mc1gt_1,145,153). coords(mc1gt_1,124,83,153).  visible(mc1gt_1,153). happensAt(walking(mc1gt_1),153).
orientation(mc1gt_0,147,154). coords(mc1gt_0,104,107,154).  visible(mc1gt_0,154). happensAt(walking(mc1gt_0),154). orientation(mc1gt_1,145,154). coords(mc1gt_1,126,83,154).  visible(mc1gt_1,154). happensAt(walking(mc1gt_1),154).
orientation(mc1gt_0,147,155). coords(mc1gt_0,104,107,155).  visible(mc1gt_0,155). happensAt(walking(mc1gt_0),155). orientation(mc1gt_1,145,155). coords(mc1gt_1,127,83,155).  visible(mc1gt_1,155). happensAt(walking(mc1gt_1),155).
orientation(mc1gt_0,147,156). coords(mc1gt_0,104,107,156).  visible(mc1gt_0,156). happensAt(walking(mc1gt_0),156). orientation(mc1gt_1,145,156). coords(mc1gt_1,128,83,156).  visible(mc1gt_1,156). happensAt(walking(mc1gt_1),156).
orientation(mc1gt_0,147,157). coords(mc1gt_0,104,107,157).  visible(mc1gt_0,157). happensAt(walking(mc1gt_0),157). orientation(mc1gt_1,145,157). coords(mc1gt_1,129,83,157).  visible(mc1gt_1,157). happensAt(walking(mc1gt_1),157).
orientation(mc1gt_0,147,158). coords(mc1gt_0,104,108,158).  visible(mc1gt_0,158). happensAt(walking(mc1gt_0),158). orientation(mc1gt_1,145,158). coords(mc1gt_1,130,84,158).  visible(mc1gt_1,158). happensAt(walking(mc1gt_1),158).
orientation(mc1gt_0,147,159). coords(mc1gt_0,105,108,159).  visible(mc1gt_0,159). happensAt(walking(mc1gt_0),159). orientation(mc1gt_1,145,159). coords(mc1gt_1,130,84,159).  visible(mc1gt_1,159). happensAt(walking(mc1gt_1),159).
orientation(mc1gt_0,147,160). coords(mc1gt_0,106,109,160).  visible(mc1gt_0,160). happensAt(walking(mc1gt_0),160). orientation(mc1gt_1,145,160). coords(mc1gt_1,130,84,160).  visible(mc1gt_1,160). happensAt(walking(mc1gt_1),160).
#show holdsAt/2.