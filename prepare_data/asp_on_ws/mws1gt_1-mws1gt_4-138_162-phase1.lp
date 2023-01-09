
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

	orientation(mws1gt_1,155,138). coords(mws1gt_1,129,153,138).  visible(mws1gt_1,138). happensAt(walking(mws1gt_1),138). orientation(mws1gt_4,0,138). coords(mws1gt_4,27,138,138).  visible(mws1gt_4,138). happensAt(walking(mws1gt_4),138).
orientation(mws1gt_1,155,139). coords(mws1gt_1,129,153,139).  visible(mws1gt_1,139). happensAt(walking(mws1gt_1),139). orientation(mws1gt_4,0,139). coords(mws1gt_4,27,137,139).  visible(mws1gt_4,139). happensAt(walking(mws1gt_4),139).
orientation(mws1gt_1,155,140). coords(mws1gt_1,129,153,140).  visible(mws1gt_1,140). happensAt(walking(mws1gt_1),140). orientation(mws1gt_4,0,140). coords(mws1gt_4,28,136,140).  visible(mws1gt_4,140). happensAt(walking(mws1gt_4),140).
orientation(mws1gt_1,155,141). coords(mws1gt_1,130,152,141).  visible(mws1gt_1,141). happensAt(walking(mws1gt_1),141). orientation(mws1gt_4,0,141). coords(mws1gt_4,28,136,141).  visible(mws1gt_4,141). happensAt(walking(mws1gt_4),141).
orientation(mws1gt_1,155,142). coords(mws1gt_1,130,152,142).  visible(mws1gt_1,142). happensAt(walking(mws1gt_1),142). orientation(mws1gt_4,0,142). coords(mws1gt_4,28,134,142).  visible(mws1gt_4,142). happensAt(walking(mws1gt_4),142).
orientation(mws1gt_1,155,143). coords(mws1gt_1,130,151,143).  visible(mws1gt_1,143). happensAt(walking(mws1gt_1),143). orientation(mws1gt_4,0,143). coords(mws1gt_4,28,134,143).  visible(mws1gt_4,143). happensAt(walking(mws1gt_4),143).
orientation(mws1gt_1,155,144). coords(mws1gt_1,131,151,144).  visible(mws1gt_1,144). happensAt(walking(mws1gt_1),144). orientation(mws1gt_4,0,144). coords(mws1gt_4,28,134,144).  visible(mws1gt_4,144). happensAt(walking(mws1gt_4),144).
orientation(mws1gt_1,155,145). coords(mws1gt_1,132,151,145).  visible(mws1gt_1,145). happensAt(walking(mws1gt_1),145). orientation(mws1gt_4,0,145). coords(mws1gt_4,28,134,145).  visible(mws1gt_4,145). happensAt(walking(mws1gt_4),145).
orientation(mws1gt_1,155,146). coords(mws1gt_1,134,151,146).  visible(mws1gt_1,146). happensAt(walking(mws1gt_1),146). orientation(mws1gt_4,0,146). coords(mws1gt_4,28,134,146).  visible(mws1gt_4,146). happensAt(walking(mws1gt_4),146).
orientation(mws1gt_1,155,147). coords(mws1gt_1,135,151,147).  visible(mws1gt_1,147). happensAt(walking(mws1gt_1),147). orientation(mws1gt_4,0,147). coords(mws1gt_4,28,134,147).  visible(mws1gt_4,147). happensAt(walking(mws1gt_4),147).
orientation(mws1gt_1,155,148). coords(mws1gt_1,136,151,148).  visible(mws1gt_1,148). happensAt(walking(mws1gt_1),148). orientation(mws1gt_4,0,148). coords(mws1gt_4,28,133,148).  visible(mws1gt_4,148). happensAt(walking(mws1gt_4),148).
orientation(mws1gt_1,155,149). coords(mws1gt_1,137,150,149).  visible(mws1gt_1,149). happensAt(walking(mws1gt_1),149). orientation(mws1gt_4,0,149). coords(mws1gt_4,28,133,149).  visible(mws1gt_4,149). happensAt(walking(mws1gt_4),149).
orientation(mws1gt_1,155,150). coords(mws1gt_1,138,150,150).  visible(mws1gt_1,150). happensAt(walking(mws1gt_1),150). orientation(mws1gt_4,0,150). coords(mws1gt_4,28,133,150).  visible(mws1gt_4,150). happensAt(walking(mws1gt_4),150).
orientation(mws1gt_1,155,151). coords(mws1gt_1,139,149,151).  visible(mws1gt_1,151). happensAt(walking(mws1gt_1),151). orientation(mws1gt_4,0,151). coords(mws1gt_4,28,132,151).  visible(mws1gt_4,151). happensAt(walking(mws1gt_4),151).
orientation(mws1gt_1,155,152). coords(mws1gt_1,139,149,152).  visible(mws1gt_1,152). happensAt(walking(mws1gt_1),152). orientation(mws1gt_4,0,152). coords(mws1gt_4,28,132,152).  visible(mws1gt_4,152). happensAt(walking(mws1gt_4),152).
orientation(mws1gt_1,155,153). coords(mws1gt_1,140,148,153).  visible(mws1gt_1,153). happensAt(walking(mws1gt_1),153). orientation(mws1gt_4,0,153). coords(mws1gt_4,28,131,153).  visible(mws1gt_4,153). happensAt(walking(mws1gt_4),153).
orientation(mws1gt_1,155,154). coords(mws1gt_1,140,148,154).  visible(mws1gt_1,154). happensAt(walking(mws1gt_1),154). orientation(mws1gt_4,0,154). coords(mws1gt_4,28,129,154).  visible(mws1gt_4,154). happensAt(walking(mws1gt_4),154).
orientation(mws1gt_1,155,155). coords(mws1gt_1,140,147,155).  visible(mws1gt_1,155). happensAt(walking(mws1gt_1),155). orientation(mws1gt_4,0,155). coords(mws1gt_4,28,129,155).  visible(mws1gt_4,155). happensAt(walking(mws1gt_4),155).
orientation(mws1gt_1,155,156). coords(mws1gt_1,141,147,156).  visible(mws1gt_1,156). happensAt(walking(mws1gt_1),156). orientation(mws1gt_4,0,156). coords(mws1gt_4,28,127,156).  visible(mws1gt_4,156). happensAt(walking(mws1gt_4),156).
orientation(mws1gt_1,155,157). coords(mws1gt_1,141,146,157).  visible(mws1gt_1,157). happensAt(walking(mws1gt_1),157). orientation(mws1gt_4,0,157). coords(mws1gt_4,28,125,157).  visible(mws1gt_4,157). happensAt(walking(mws1gt_4),157).
orientation(mws1gt_1,155,158). coords(mws1gt_1,141,145,158).  visible(mws1gt_1,158). happensAt(walking(mws1gt_1),158). orientation(mws1gt_4,0,158). coords(mws1gt_4,28,124,158).  visible(mws1gt_4,158). happensAt(walking(mws1gt_4),158).
orientation(mws1gt_1,155,159). coords(mws1gt_1,141,144,159).  visible(mws1gt_1,159). happensAt(walking(mws1gt_1),159). orientation(mws1gt_4,0,159). coords(mws1gt_4,28,124,159).  visible(mws1gt_4,159). happensAt(walking(mws1gt_4),159).
orientation(mws1gt_1,155,160). coords(mws1gt_1,142,143,160).  visible(mws1gt_1,160). happensAt(walking(mws1gt_1),160). orientation(mws1gt_4,0,160). coords(mws1gt_4,28,124,160).  visible(mws1gt_4,160). happensAt(walking(mws1gt_4),160).
orientation(mws1gt_1,155,161). coords(mws1gt_1,142,143,161).  visible(mws1gt_1,161). happensAt(walking(mws1gt_1),161). orientation(mws1gt_4,0,161). coords(mws1gt_4,28,124,161).  visible(mws1gt_4,161). happensAt(walking(mws1gt_4),161).
orientation(mws1gt_1,155,162). coords(mws1gt_1,142,143,162).  visible(mws1gt_1,162). happensAt(walking(mws1gt_1),162). orientation(mws1gt_4,0,162). coords(mws1gt_4,28,124,162).  visible(mws1gt_4,162). happensAt(walking(mws1gt_4),162).
#show holdsAt/2.