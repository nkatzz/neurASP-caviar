
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

	orientation(mws1gt_1,160,68). coords(mws1gt_1,72,189,68).  visible(mws1gt_1,68). happensAt(walking(mws1gt_1),68). orientation(mws1gt_3,90,68). coords(mws1gt_3,266,278,68).  happensAt(appear(mws1gt_3),68). happensAt(walking(mws1gt_3),68).
orientation(mws1gt_1,160,69). coords(mws1gt_1,72,189,69).  visible(mws1gt_1,69). happensAt(walking(mws1gt_1),69). orientation(mws1gt_3,0,69). coords(mws1gt_3,267,280,69).  visible(mws1gt_3,69). happensAt(walking(mws1gt_3),69).
orientation(mws1gt_1,160,70). coords(mws1gt_1,72,189,70).  visible(mws1gt_1,70). happensAt(walking(mws1gt_1),70). orientation(mws1gt_3,60,70). coords(mws1gt_3,266,278,70).  visible(mws1gt_3,70). happensAt(walking(mws1gt_3),70).
orientation(mws1gt_1,160,71). coords(mws1gt_1,72,187,71).  visible(mws1gt_1,71). happensAt(walking(mws1gt_1),71). orientation(mws1gt_3,69,71). coords(mws1gt_3,266,274,71).  visible(mws1gt_3,71). happensAt(walking(mws1gt_3),71).
orientation(mws1gt_1,160,72). coords(mws1gt_1,73,187,72).  visible(mws1gt_1,72). happensAt(walking(mws1gt_1),72). orientation(mws1gt_3,69,72). coords(mws1gt_3,263,273,72).  visible(mws1gt_3,72). happensAt(walking(mws1gt_3),72).
orientation(mws1gt_1,160,73). coords(mws1gt_1,74,187,73).  visible(mws1gt_1,73). happensAt(walking(mws1gt_1),73). orientation(mws1gt_3,77,73). coords(mws1gt_3,262,272,73).  visible(mws1gt_3,73). happensAt(walking(mws1gt_3),73).
orientation(mws1gt_1,160,74). coords(mws1gt_1,75,187,74).  visible(mws1gt_1,74). happensAt(walking(mws1gt_1),74). orientation(mws1gt_3,68,74). coords(mws1gt_3,259,271,74).  visible(mws1gt_3,74). happensAt(walking(mws1gt_3),74).
orientation(mws1gt_1,160,75). coords(mws1gt_1,77,187,75).  visible(mws1gt_1,75). happensAt(walking(mws1gt_1),75). orientation(mws1gt_3,56,75). coords(mws1gt_3,259,270,75).  visible(mws1gt_3,75). happensAt(walking(mws1gt_3),75).
orientation(mws1gt_1,160,76). coords(mws1gt_1,79,187,76).  visible(mws1gt_1,76). happensAt(walking(mws1gt_1),76). orientation(mws1gt_3,47,76). coords(mws1gt_3,258,270,76).  visible(mws1gt_3,76). happensAt(walking(mws1gt_3),76).
orientation(mws1gt_1,160,77). coords(mws1gt_1,79,186,77).  visible(mws1gt_1,77). happensAt(walking(mws1gt_1),77). orientation(mws1gt_3,40,77). coords(mws1gt_3,258,270,77).  visible(mws1gt_3,77). happensAt(walking(mws1gt_3),77).
orientation(mws1gt_1,160,78). coords(mws1gt_1,80,186,78).  visible(mws1gt_1,78). happensAt(walking(mws1gt_1),78). orientation(mws1gt_3,32,78). coords(mws1gt_3,257,269,78).  visible(mws1gt_3,78). happensAt(walking(mws1gt_3),78).
orientation(mws1gt_1,160,79). coords(mws1gt_1,81,185,79).  visible(mws1gt_1,79). happensAt(walking(mws1gt_1),79). orientation(mws1gt_3,32,79). coords(mws1gt_3,257,269,79).  visible(mws1gt_3,79). happensAt(walking(mws1gt_3),79).
orientation(mws1gt_1,160,80). coords(mws1gt_1,81,184,80).  visible(mws1gt_1,80). happensAt(walking(mws1gt_1),80). orientation(mws1gt_3,27,80). coords(mws1gt_3,258,268,80).  visible(mws1gt_3,80). happensAt(walking(mws1gt_3),80).
orientation(mws1gt_1,160,81). coords(mws1gt_1,81,184,81).  visible(mws1gt_1,81). happensAt(walking(mws1gt_1),81). orientation(mws1gt_3,29,81). coords(mws1gt_3,257,267,81).  visible(mws1gt_3,81). happensAt(walking(mws1gt_3),81).
orientation(mws1gt_1,160,82). coords(mws1gt_1,81,184,82).  visible(mws1gt_1,82). happensAt(walking(mws1gt_1),82). orientation(mws1gt_3,29,82). coords(mws1gt_3,256,265,82).  visible(mws1gt_3,82). happensAt(walking(mws1gt_3),82).
orientation(mws1gt_1,160,83). coords(mws1gt_1,81,184,83).  visible(mws1gt_1,83). happensAt(walking(mws1gt_1),83). orientation(mws1gt_3,26,83). coords(mws1gt_3,254,263,83).  visible(mws1gt_3,83). happensAt(walking(mws1gt_3),83).
orientation(mws1gt_1,160,84). coords(mws1gt_1,81,184,84).  visible(mws1gt_1,84). happensAt(walking(mws1gt_1),84). orientation(mws1gt_3,24,84). coords(mws1gt_3,254,261,84).  visible(mws1gt_3,84). happensAt(walking(mws1gt_3),84).
orientation(mws1gt_1,160,85). coords(mws1gt_1,81,183,85).  visible(mws1gt_1,85). happensAt(walking(mws1gt_1),85). orientation(mws1gt_3,24,85). coords(mws1gt_3,252,256,85).  visible(mws1gt_3,85). happensAt(walking(mws1gt_3),85).
orientation(mws1gt_1,160,86). coords(mws1gt_1,81,183,86).  visible(mws1gt_1,86). happensAt(walking(mws1gt_1),86). orientation(mws1gt_3,24,86). coords(mws1gt_3,250,252,86).  visible(mws1gt_3,86). happensAt(walking(mws1gt_3),86).
orientation(mws1gt_1,160,87). coords(mws1gt_1,82,182,87).  visible(mws1gt_1,87). happensAt(walking(mws1gt_1),87). orientation(mws1gt_3,24,87). coords(mws1gt_3,250,249,87).  visible(mws1gt_3,87). happensAt(walking(mws1gt_3),87).
orientation(mws1gt_1,160,88). coords(mws1gt_1,82,182,88).  visible(mws1gt_1,88). happensAt(walking(mws1gt_1),88). orientation(mws1gt_3,24,88). coords(mws1gt_3,250,246,88).  visible(mws1gt_3,88). happensAt(walking(mws1gt_3),88).
orientation(mws1gt_1,160,89). coords(mws1gt_1,85,182,89).  visible(mws1gt_1,89). happensAt(walking(mws1gt_1),89). orientation(mws1gt_3,24,89). coords(mws1gt_3,250,245,89).  visible(mws1gt_3,89). happensAt(walking(mws1gt_3),89).
orientation(mws1gt_1,160,90). coords(mws1gt_1,85,182,90).  visible(mws1gt_1,90). happensAt(walking(mws1gt_1),90). orientation(mws1gt_3,24,90). coords(mws1gt_3,249,245,90).  visible(mws1gt_3,90). happensAt(walking(mws1gt_3),90).
orientation(mws1gt_1,160,91). coords(mws1gt_1,86,181,91).  visible(mws1gt_1,91). happensAt(walking(mws1gt_1),91). orientation(mws1gt_3,24,91). coords(mws1gt_3,248,245,91).  visible(mws1gt_3,91). happensAt(walking(mws1gt_3),91).
orientation(mws1gt_1,160,92). coords(mws1gt_1,88,181,92).  visible(mws1gt_1,92). happensAt(walking(mws1gt_1),92). orientation(mws1gt_3,24,92). coords(mws1gt_3,248,244,92).  visible(mws1gt_3,92). happensAt(walking(mws1gt_3),92).
#show holdsAt/2.