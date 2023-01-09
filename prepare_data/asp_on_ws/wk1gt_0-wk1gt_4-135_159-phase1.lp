
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

	orientation(wk1gt_0,165,135). coords(wk1gt_0,229,156,135).  visible(wk1gt_0,135). happensAt(walking(wk1gt_0),135). orientation(wk1gt_4,0,135). coords(wk1gt_4,45,185,135).  visible(wk1gt_4,135). happensAt(walking(wk1gt_4),135).
orientation(wk1gt_0,165,136). coords(wk1gt_0,229,158,136).  visible(wk1gt_0,136). happensAt(walking(wk1gt_0),136). orientation(wk1gt_4,0,136). coords(wk1gt_4,45,185,136).  visible(wk1gt_4,136). happensAt(walking(wk1gt_4),136).
orientation(wk1gt_0,165,137). coords(wk1gt_0,232,159,137).  visible(wk1gt_0,137). happensAt(walking(wk1gt_0),137). orientation(wk1gt_4,0,137). coords(wk1gt_4,45,184,137).  visible(wk1gt_4,137). happensAt(walking(wk1gt_4),137).
orientation(wk1gt_0,178,138). coords(wk1gt_0,234,162,138).  visible(wk1gt_0,138). happensAt(walking(wk1gt_0),138). orientation(wk1gt_4,0,138). coords(wk1gt_4,45,184,138).  visible(wk1gt_4,138). happensAt(walking(wk1gt_4),138).
orientation(wk1gt_0,7,139). coords(wk1gt_0,237,164,139).  visible(wk1gt_0,139). happensAt(walking(wk1gt_0),139). orientation(wk1gt_4,0,139). coords(wk1gt_4,45,184,139).  visible(wk1gt_4,139). happensAt(walking(wk1gt_4),139).
orientation(wk1gt_0,10,140). coords(wk1gt_0,240,167,140).  visible(wk1gt_0,140). happensAt(walking(wk1gt_0),140). orientation(wk1gt_4,0,140). coords(wk1gt_4,45,184,140).  visible(wk1gt_4,140). happensAt(walking(wk1gt_4),140).
orientation(wk1gt_0,2,141). coords(wk1gt_0,242,172,141).  visible(wk1gt_0,141). happensAt(walking(wk1gt_0),141). orientation(wk1gt_4,0,141). coords(wk1gt_4,45,184,141).  visible(wk1gt_4,141). happensAt(walking(wk1gt_4),141).
orientation(wk1gt_0,171,142). coords(wk1gt_0,244,178,142).  visible(wk1gt_0,142). happensAt(walking(wk1gt_0),142). orientation(wk1gt_4,0,142). coords(wk1gt_4,46,184,142).  visible(wk1gt_4,142). happensAt(walking(wk1gt_4),142).
orientation(wk1gt_0,170,143). coords(wk1gt_0,245,181,143).  visible(wk1gt_0,143). happensAt(walking(wk1gt_0),143). orientation(wk1gt_4,0,143). coords(wk1gt_4,46,184,143).  visible(wk1gt_4,143). happensAt(walking(wk1gt_4),143).
orientation(wk1gt_0,167,144). coords(wk1gt_0,247,185,144).  visible(wk1gt_0,144). happensAt(walking(wk1gt_0),144). orientation(wk1gt_4,0,144). coords(wk1gt_4,46,184,144).  visible(wk1gt_4,144). happensAt(walking(wk1gt_4),144).
orientation(wk1gt_0,175,145). coords(wk1gt_0,248,188,145).  visible(wk1gt_0,145). happensAt(walking(wk1gt_0),145). orientation(wk1gt_4,0,145). coords(wk1gt_4,46,184,145).  visible(wk1gt_4,145). happensAt(walking(wk1gt_4),145).
orientation(wk1gt_0,175,146). coords(wk1gt_0,249,191,146).  visible(wk1gt_0,146). happensAt(walking(wk1gt_0),146). orientation(wk1gt_4,0,146). coords(wk1gt_4,46,184,146).  visible(wk1gt_4,146). happensAt(walking(wk1gt_4),146).
orientation(wk1gt_0,179,147). coords(wk1gt_0,251,195,147).  visible(wk1gt_0,147). happensAt(walking(wk1gt_0),147). orientation(wk1gt_4,0,147). coords(wk1gt_4,46,183,147).  visible(wk1gt_4,147). happensAt(walking(wk1gt_4),147).
orientation(wk1gt_0,178,148). coords(wk1gt_0,253,201,148).  visible(wk1gt_0,148). happensAt(walking(wk1gt_0),148). orientation(wk1gt_4,0,148). coords(wk1gt_4,46,183,148).  visible(wk1gt_4,148). happensAt(walking(wk1gt_4),148).
orientation(wk1gt_0,178,149). coords(wk1gt_0,253,205,149).  visible(wk1gt_0,149). happensAt(walking(wk1gt_0),149). orientation(wk1gt_4,0,149). coords(wk1gt_4,47,183,149).  visible(wk1gt_4,149). happensAt(walking(wk1gt_4),149).
orientation(wk1gt_0,0,150). coords(wk1gt_0,253,209,150).  visible(wk1gt_0,150). happensAt(walking(wk1gt_0),150). orientation(wk1gt_4,163,150). coords(wk1gt_4,47,183,150).  visible(wk1gt_4,150). happensAt(walking(wk1gt_4),150).
orientation(wk1gt_0,0,151). coords(wk1gt_0,254,213,151).  visible(wk1gt_0,151). happensAt(walking(wk1gt_0),151). orientation(wk1gt_4,166,151). coords(wk1gt_4,47,183,151).  visible(wk1gt_4,151). happensAt(walking(wk1gt_4),151).
orientation(wk1gt_0,0,152). coords(wk1gt_0,255,217,152).  visible(wk1gt_0,152). happensAt(walking(wk1gt_0),152). orientation(wk1gt_4,162,152). coords(wk1gt_4,47,183,152).  visible(wk1gt_4,152). happensAt(walking(wk1gt_4),152).
orientation(wk1gt_0,0,153). coords(wk1gt_0,256,219,153).  visible(wk1gt_0,153). happensAt(walking(wk1gt_0),153). orientation(wk1gt_4,158,153). coords(wk1gt_4,47,183,153).  visible(wk1gt_4,153). happensAt(walking(wk1gt_4),153).
orientation(wk1gt_0,12,154). coords(wk1gt_0,257,221,154).  visible(wk1gt_0,154). happensAt(walking(wk1gt_0),154). orientation(wk1gt_4,163,154). coords(wk1gt_4,47,183,154).  visible(wk1gt_4,154). happensAt(walking(wk1gt_4),154).
orientation(wk1gt_0,28,155). coords(wk1gt_0,260,223,155).  visible(wk1gt_0,155). happensAt(walking(wk1gt_0),155). orientation(wk1gt_4,157,155). coords(wk1gt_4,47,183,155).  visible(wk1gt_4,155). happensAt(walking(wk1gt_4),155).
orientation(wk1gt_0,25,156). coords(wk1gt_0,262,227,156).  visible(wk1gt_0,156). happensAt(walking(wk1gt_0),156). orientation(wk1gt_4,164,156). coords(wk1gt_4,47,183,156).  visible(wk1gt_4,156). happensAt(walking(wk1gt_4),156).
orientation(wk1gt_0,22,157). coords(wk1gt_0,264,234,157).  visible(wk1gt_0,157). happensAt(walking(wk1gt_0),157). orientation(wk1gt_4,164,157). coords(wk1gt_4,47,183,157).  visible(wk1gt_4,157). happensAt(walking(wk1gt_4),157).
orientation(wk1gt_0,12,158). coords(wk1gt_0,266,240,158).  visible(wk1gt_0,158). happensAt(walking(wk1gt_0),158). orientation(wk1gt_4,160,158). coords(wk1gt_4,47,183,158).  visible(wk1gt_4,158). happensAt(walking(wk1gt_4),158).
orientation(wk1gt_0,23,159). coords(wk1gt_0,268,244,159).  visible(wk1gt_0,159). happensAt(walking(wk1gt_0),159). orientation(wk1gt_4,160,159). coords(wk1gt_4,47,183,159).  visible(wk1gt_4,159). happensAt(walking(wk1gt_4),159).
#show holdsAt/2.