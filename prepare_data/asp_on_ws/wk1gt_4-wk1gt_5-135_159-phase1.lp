
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

	orientation(wk1gt_4,0,135). coords(wk1gt_4,45,185,135).  visible(wk1gt_4,135). happensAt(walking(wk1gt_4),135). orientation(wk1gt_5,169,135). coords(wk1gt_5,44,169,135).  visible(wk1gt_5,135). happensAt(walking(wk1gt_5),135).
orientation(wk1gt_4,0,136). coords(wk1gt_4,45,185,136).  visible(wk1gt_4,136). happensAt(walking(wk1gt_4),136). orientation(wk1gt_5,169,136). coords(wk1gt_5,44,169,136).  visible(wk1gt_5,136). happensAt(walking(wk1gt_5),136).
orientation(wk1gt_4,0,137). coords(wk1gt_4,45,184,137).  visible(wk1gt_4,137). happensAt(walking(wk1gt_4),137). orientation(wk1gt_5,169,137). coords(wk1gt_5,44,169,137).  visible(wk1gt_5,137). happensAt(walking(wk1gt_5),137).
orientation(wk1gt_4,0,138). coords(wk1gt_4,45,184,138).  visible(wk1gt_4,138). happensAt(walking(wk1gt_4),138). orientation(wk1gt_5,169,138). coords(wk1gt_5,44,169,138).  visible(wk1gt_5,138). happensAt(walking(wk1gt_5),138).
orientation(wk1gt_4,0,139). coords(wk1gt_4,45,184,139).  visible(wk1gt_4,139). happensAt(walking(wk1gt_4),139). orientation(wk1gt_5,169,139). coords(wk1gt_5,44,169,139).  visible(wk1gt_5,139). happensAt(walking(wk1gt_5),139).
orientation(wk1gt_4,0,140). coords(wk1gt_4,45,184,140).  visible(wk1gt_4,140). happensAt(walking(wk1gt_4),140). orientation(wk1gt_5,169,140). coords(wk1gt_5,44,168,140).  visible(wk1gt_5,140). happensAt(walking(wk1gt_5),140).
orientation(wk1gt_4,0,141). coords(wk1gt_4,45,184,141).  visible(wk1gt_4,141). happensAt(walking(wk1gt_4),141). orientation(wk1gt_5,169,141). coords(wk1gt_5,45,168,141).  visible(wk1gt_5,141). happensAt(walking(wk1gt_5),141).
orientation(wk1gt_4,0,142). coords(wk1gt_4,46,184,142).  visible(wk1gt_4,142). happensAt(walking(wk1gt_4),142). orientation(wk1gt_5,169,142). coords(wk1gt_5,45,167,142).  visible(wk1gt_5,142). happensAt(walking(wk1gt_5),142).
orientation(wk1gt_4,0,143). coords(wk1gt_4,46,184,143).  visible(wk1gt_4,143). happensAt(walking(wk1gt_4),143). orientation(wk1gt_5,169,143). coords(wk1gt_5,45,167,143).  visible(wk1gt_5,143). happensAt(walking(wk1gt_5),143).
orientation(wk1gt_4,0,144). coords(wk1gt_4,46,184,144).  visible(wk1gt_4,144). happensAt(walking(wk1gt_4),144). orientation(wk1gt_5,169,144). coords(wk1gt_5,45,167,144).  visible(wk1gt_5,144). happensAt(walking(wk1gt_5),144).
orientation(wk1gt_4,0,145). coords(wk1gt_4,46,184,145).  visible(wk1gt_4,145). happensAt(walking(wk1gt_4),145). orientation(wk1gt_5,169,145). coords(wk1gt_5,45,167,145).  visible(wk1gt_5,145). happensAt(walking(wk1gt_5),145).
orientation(wk1gt_4,0,146). coords(wk1gt_4,46,184,146).  visible(wk1gt_4,146). happensAt(walking(wk1gt_4),146). orientation(wk1gt_5,162,146). coords(wk1gt_5,45,167,146).  visible(wk1gt_5,146). happensAt(walking(wk1gt_5),146).
orientation(wk1gt_4,0,147). coords(wk1gt_4,46,183,147).  visible(wk1gt_4,147). happensAt(walking(wk1gt_4),147). orientation(wk1gt_5,162,147). coords(wk1gt_5,45,167,147).  visible(wk1gt_5,147). happensAt(walking(wk1gt_5),147).
orientation(wk1gt_4,0,148). coords(wk1gt_4,46,183,148).  visible(wk1gt_4,148). happensAt(walking(wk1gt_4),148). orientation(wk1gt_5,162,148). coords(wk1gt_5,45,167,148).  visible(wk1gt_5,148). happensAt(walking(wk1gt_5),148).
orientation(wk1gt_4,0,149). coords(wk1gt_4,47,183,149).  visible(wk1gt_4,149). happensAt(walking(wk1gt_4),149). orientation(wk1gt_5,162,149). coords(wk1gt_5,45,167,149).  visible(wk1gt_5,149). happensAt(walking(wk1gt_5),149).
orientation(wk1gt_4,163,150). coords(wk1gt_4,47,183,150).  visible(wk1gt_4,150). happensAt(walking(wk1gt_4),150). orientation(wk1gt_5,162,150). coords(wk1gt_5,45,167,150).  visible(wk1gt_5,150). happensAt(walking(wk1gt_5),150).
orientation(wk1gt_4,166,151). coords(wk1gt_4,47,183,151).  visible(wk1gt_4,151). happensAt(walking(wk1gt_4),151). orientation(wk1gt_5,162,151). coords(wk1gt_5,45,167,151).  visible(wk1gt_5,151). happensAt(walking(wk1gt_5),151).
orientation(wk1gt_4,162,152). coords(wk1gt_4,47,183,152).  visible(wk1gt_4,152). happensAt(walking(wk1gt_4),152). orientation(wk1gt_5,162,152). coords(wk1gt_5,45,167,152).  visible(wk1gt_5,152). happensAt(walking(wk1gt_5),152).
orientation(wk1gt_4,158,153). coords(wk1gt_4,47,183,153).  visible(wk1gt_4,153). happensAt(walking(wk1gt_4),153). orientation(wk1gt_5,162,153). coords(wk1gt_5,45,167,153).  visible(wk1gt_5,153). happensAt(walking(wk1gt_5),153).
orientation(wk1gt_4,163,154). coords(wk1gt_4,47,183,154).  visible(wk1gt_4,154). happensAt(walking(wk1gt_4),154). orientation(wk1gt_5,162,154). coords(wk1gt_5,45,167,154).  visible(wk1gt_5,154). happensAt(walking(wk1gt_5),154).
orientation(wk1gt_4,157,155). coords(wk1gt_4,47,183,155).  visible(wk1gt_4,155). happensAt(walking(wk1gt_4),155). orientation(wk1gt_5,162,155). coords(wk1gt_5,45,167,155).  visible(wk1gt_5,155). happensAt(active(wk1gt_5),155).
orientation(wk1gt_4,164,156). coords(wk1gt_4,47,183,156).  visible(wk1gt_4,156). happensAt(walking(wk1gt_4),156). orientation(wk1gt_5,162,156). coords(wk1gt_5,46,166,156).  visible(wk1gt_5,156). happensAt(active(wk1gt_5),156).
orientation(wk1gt_4,164,157). coords(wk1gt_4,47,183,157).  visible(wk1gt_4,157). happensAt(walking(wk1gt_4),157). orientation(wk1gt_5,162,157). coords(wk1gt_5,46,166,157).  visible(wk1gt_5,157). happensAt(active(wk1gt_5),157).
orientation(wk1gt_4,160,158). coords(wk1gt_4,47,183,158).  visible(wk1gt_4,158). happensAt(walking(wk1gt_4),158). orientation(wk1gt_5,162,158). coords(wk1gt_5,46,166,158).  visible(wk1gt_5,158). happensAt(active(wk1gt_5),158).
orientation(wk1gt_4,160,159). coords(wk1gt_4,47,183,159).  visible(wk1gt_4,159). happensAt(walking(wk1gt_4),159). orientation(wk1gt_5,162,159). coords(wk1gt_5,46,166,159).  visible(wk1gt_5,159). happensAt(active(wk1gt_5),159).
#show holdsAt/2.