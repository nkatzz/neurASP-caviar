
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

	orientation(mc1gt_1,145,69). coords(mc1gt_1,94,83,69).  visible(mc1gt_1,69). happensAt(walking(mc1gt_1),69). orientation(mc1gt_2,144,69). coords(mc1gt_2,78,65,69).  visible(mc1gt_2,69). happensAt(walking(mc1gt_2),69).
orientation(mc1gt_1,145,70). coords(mc1gt_1,94,83,70).  visible(mc1gt_1,70). happensAt(walking(mc1gt_1),70). orientation(mc1gt_2,144,70). coords(mc1gt_2,78,65,70).  visible(mc1gt_2,70). happensAt(walking(mc1gt_2),70).
orientation(mc1gt_1,145,71). coords(mc1gt_1,94,83,71).  visible(mc1gt_1,71). happensAt(walking(mc1gt_1),71). orientation(mc1gt_2,144,71). coords(mc1gt_2,78,65,71).  visible(mc1gt_2,71). happensAt(walking(mc1gt_2),71).
orientation(mc1gt_1,145,72). coords(mc1gt_1,94,83,72).  visible(mc1gt_1,72). happensAt(walking(mc1gt_1),72). orientation(mc1gt_2,144,72). coords(mc1gt_2,78,65,72).  visible(mc1gt_2,72). happensAt(walking(mc1gt_2),72).
orientation(mc1gt_1,145,73). coords(mc1gt_1,94,83,73).  visible(mc1gt_1,73). happensAt(walking(mc1gt_1),73). orientation(mc1gt_2,144,73). coords(mc1gt_2,78,65,73).  visible(mc1gt_2,73). happensAt(walking(mc1gt_2),73).
orientation(mc1gt_1,145,74). coords(mc1gt_1,94,83,74).  visible(mc1gt_1,74). happensAt(walking(mc1gt_1),74). orientation(mc1gt_2,144,74). coords(mc1gt_2,78,65,74).  visible(mc1gt_2,74). happensAt(walking(mc1gt_2),74).
orientation(mc1gt_1,145,75). coords(mc1gt_1,94,83,75).  visible(mc1gt_1,75). happensAt(walking(mc1gt_1),75). orientation(mc1gt_2,144,75). coords(mc1gt_2,78,65,75).  visible(mc1gt_2,75). happensAt(walking(mc1gt_2),75).
orientation(mc1gt_1,145,76). coords(mc1gt_1,94,83,76).  visible(mc1gt_1,76). happensAt(walking(mc1gt_1),76). orientation(mc1gt_2,144,76). coords(mc1gt_2,78,65,76).  visible(mc1gt_2,76). happensAt(walking(mc1gt_2),76).
orientation(mc1gt_1,145,77). coords(mc1gt_1,94,83,77).  visible(mc1gt_1,77). happensAt(walking(mc1gt_1),77). orientation(mc1gt_2,144,77). coords(mc1gt_2,78,65,77).  visible(mc1gt_2,77). happensAt(walking(mc1gt_2),77).
orientation(mc1gt_1,145,78). coords(mc1gt_1,94,83,78).  visible(mc1gt_1,78). happensAt(walking(mc1gt_1),78). orientation(mc1gt_2,144,78). coords(mc1gt_2,78,65,78).  visible(mc1gt_2,78). happensAt(walking(mc1gt_2),78).
orientation(mc1gt_1,145,79). coords(mc1gt_1,94,83,79).  visible(mc1gt_1,79). happensAt(walking(mc1gt_1),79). orientation(mc1gt_2,144,79). coords(mc1gt_2,78,65,79).  visible(mc1gt_2,79). happensAt(walking(mc1gt_2),79).
orientation(mc1gt_1,145,80). coords(mc1gt_1,94,83,80).  visible(mc1gt_1,80). happensAt(walking(mc1gt_1),80). orientation(mc1gt_2,144,80). coords(mc1gt_2,78,65,80).  visible(mc1gt_2,80). happensAt(walking(mc1gt_2),80).
orientation(mc1gt_1,145,81). coords(mc1gt_1,94,83,81).  visible(mc1gt_1,81). happensAt(walking(mc1gt_1),81). orientation(mc1gt_2,144,81). coords(mc1gt_2,78,65,81).  visible(mc1gt_2,81). happensAt(walking(mc1gt_2),81).
orientation(mc1gt_1,145,82). coords(mc1gt_1,94,83,82).  visible(mc1gt_1,82). happensAt(walking(mc1gt_1),82). orientation(mc1gt_2,144,82). coords(mc1gt_2,78,65,82).  visible(mc1gt_2,82). happensAt(walking(mc1gt_2),82).
orientation(mc1gt_1,145,83). coords(mc1gt_1,94,83,83).  visible(mc1gt_1,83). happensAt(walking(mc1gt_1),83). orientation(mc1gt_2,144,83). coords(mc1gt_2,78,65,83).  visible(mc1gt_2,83). happensAt(walking(mc1gt_2),83).
orientation(mc1gt_1,145,84). coords(mc1gt_1,94,83,84).  visible(mc1gt_1,84). happensAt(walking(mc1gt_1),84). orientation(mc1gt_2,144,84). coords(mc1gt_2,78,65,84).  visible(mc1gt_2,84). happensAt(walking(mc1gt_2),84).
orientation(mc1gt_1,145,85). coords(mc1gt_1,94,83,85).  visible(mc1gt_1,85). happensAt(walking(mc1gt_1),85). orientation(mc1gt_2,144,85). coords(mc1gt_2,78,65,85).  visible(mc1gt_2,85). happensAt(walking(mc1gt_2),85).
orientation(mc1gt_1,145,86). coords(mc1gt_1,94,83,86).  visible(mc1gt_1,86). happensAt(walking(mc1gt_1),86). orientation(mc1gt_2,144,86). coords(mc1gt_2,78,65,86).  visible(mc1gt_2,86). happensAt(walking(mc1gt_2),86).
orientation(mc1gt_1,145,87). coords(mc1gt_1,94,83,87).  visible(mc1gt_1,87). happensAt(walking(mc1gt_1),87). orientation(mc1gt_2,144,87). coords(mc1gt_2,78,65,87).  visible(mc1gt_2,87). happensAt(walking(mc1gt_2),87).
orientation(mc1gt_1,145,88). coords(mc1gt_1,94,83,88).  visible(mc1gt_1,88). happensAt(walking(mc1gt_1),88). orientation(mc1gt_2,144,88). coords(mc1gt_2,78,65,88).  visible(mc1gt_2,88). happensAt(walking(mc1gt_2),88).
orientation(mc1gt_1,145,89). coords(mc1gt_1,94,83,89).  visible(mc1gt_1,89). happensAt(walking(mc1gt_1),89). orientation(mc1gt_2,144,89). coords(mc1gt_2,78,65,89).  visible(mc1gt_2,89). happensAt(walking(mc1gt_2),89).
orientation(mc1gt_1,145,90). coords(mc1gt_1,94,83,90).  visible(mc1gt_1,90). happensAt(walking(mc1gt_1),90). orientation(mc1gt_2,144,90). coords(mc1gt_2,78,65,90).  visible(mc1gt_2,90). happensAt(walking(mc1gt_2),90).
orientation(mc1gt_1,145,91). coords(mc1gt_1,94,83,91).  visible(mc1gt_1,91). happensAt(walking(mc1gt_1),91). orientation(mc1gt_2,144,91). coords(mc1gt_2,78,65,91).  visible(mc1gt_2,91). happensAt(walking(mc1gt_2),91).
orientation(mc1gt_1,145,92). coords(mc1gt_1,94,83,92).  visible(mc1gt_1,92). happensAt(walking(mc1gt_1),92). orientation(mc1gt_2,144,92). coords(mc1gt_2,78,65,92).  visible(mc1gt_2,92). happensAt(walking(mc1gt_2),92).
orientation(mc1gt_1,145,93). coords(mc1gt_1,94,83,93).  visible(mc1gt_1,93). happensAt(walking(mc1gt_1),93). orientation(mc1gt_2,144,93). coords(mc1gt_2,78,65,93).  visible(mc1gt_2,93). happensAt(walking(mc1gt_2),93).
#show holdsAt/2.