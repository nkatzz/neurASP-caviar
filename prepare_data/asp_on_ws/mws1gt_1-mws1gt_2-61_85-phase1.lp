
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

	orientation(mws1gt_1,172,61). coords(mws1gt_1,71,192,61).  visible(mws1gt_1,61). happensAt(walking(mws1gt_1),61). orientation(mws1gt_2,165,61). coords(mws1gt_2,39,157,61).  visible(mws1gt_2,61). happensAt(walking(mws1gt_2),61).
orientation(mws1gt_1,172,62). coords(mws1gt_1,71,192,62).  visible(mws1gt_1,62). happensAt(walking(mws1gt_1),62). orientation(mws1gt_2,165,62). coords(mws1gt_2,39,158,62).  visible(mws1gt_2,62). happensAt(walking(mws1gt_2),62).
orientation(mws1gt_1,172,63). coords(mws1gt_1,72,191,63).  visible(mws1gt_1,63). happensAt(walking(mws1gt_1),63). orientation(mws1gt_2,165,63). coords(mws1gt_2,39,158,63).  visible(mws1gt_2,63). happensAt(walking(mws1gt_2),63).
orientation(mws1gt_1,172,64). coords(mws1gt_1,72,191,64).  visible(mws1gt_1,64). happensAt(walking(mws1gt_1),64). orientation(mws1gt_2,165,64). coords(mws1gt_2,38,158,64).  visible(mws1gt_2,64). happensAt(walking(mws1gt_2),64).
orientation(mws1gt_1,172,65). coords(mws1gt_1,72,191,65).  visible(mws1gt_1,65). happensAt(walking(mws1gt_1),65). orientation(mws1gt_2,165,65). coords(mws1gt_2,38,159,65).  visible(mws1gt_2,65). happensAt(walking(mws1gt_2),65).
orientation(mws1gt_1,172,66). coords(mws1gt_1,72,191,66).  visible(mws1gt_1,66). happensAt(walking(mws1gt_1),66). orientation(mws1gt_2,165,66). coords(mws1gt_2,38,159,66).  visible(mws1gt_2,66). happensAt(walking(mws1gt_2),66).
orientation(mws1gt_1,160,67). coords(mws1gt_1,72,190,67).  visible(mws1gt_1,67). happensAt(walking(mws1gt_1),67). orientation(mws1gt_2,165,67). coords(mws1gt_2,37,160,67).  visible(mws1gt_2,67). happensAt(walking(mws1gt_2),67).
orientation(mws1gt_1,160,68). coords(mws1gt_1,72,189,68).  visible(mws1gt_1,68). happensAt(walking(mws1gt_1),68). orientation(mws1gt_2,165,68). coords(mws1gt_2,37,161,68).  visible(mws1gt_2,68). happensAt(walking(mws1gt_2),68).
orientation(mws1gt_1,160,69). coords(mws1gt_1,72,189,69).  visible(mws1gt_1,69). happensAt(walking(mws1gt_1),69). orientation(mws1gt_2,162,69). coords(mws1gt_2,37,162,69).  visible(mws1gt_2,69). happensAt(walking(mws1gt_2),69).
orientation(mws1gt_1,160,70). coords(mws1gt_1,72,189,70).  visible(mws1gt_1,70). happensAt(walking(mws1gt_1),70). orientation(mws1gt_2,162,70). coords(mws1gt_2,37,162,70).  visible(mws1gt_2,70). happensAt(walking(mws1gt_2),70).
orientation(mws1gt_1,160,71). coords(mws1gt_1,72,187,71).  visible(mws1gt_1,71). happensAt(walking(mws1gt_1),71). orientation(mws1gt_2,162,71). coords(mws1gt_2,37,164,71).  visible(mws1gt_2,71). happensAt(walking(mws1gt_2),71).
orientation(mws1gt_1,160,72). coords(mws1gt_1,73,187,72).  visible(mws1gt_1,72). happensAt(walking(mws1gt_1),72). orientation(mws1gt_2,162,72). coords(mws1gt_2,37,166,72).  visible(mws1gt_2,72). happensAt(walking(mws1gt_2),72).
orientation(mws1gt_1,160,73). coords(mws1gt_1,74,187,73).  visible(mws1gt_1,73). happensAt(walking(mws1gt_1),73). orientation(mws1gt_2,162,73). coords(mws1gt_2,37,167,73).  visible(mws1gt_2,73). happensAt(walking(mws1gt_2),73).
orientation(mws1gt_1,160,74). coords(mws1gt_1,75,187,74).  visible(mws1gt_1,74). happensAt(walking(mws1gt_1),74). orientation(mws1gt_2,162,74). coords(mws1gt_2,37,168,74).  visible(mws1gt_2,74). happensAt(walking(mws1gt_2),74).
orientation(mws1gt_1,160,75). coords(mws1gt_1,77,187,75).  visible(mws1gt_1,75). happensAt(walking(mws1gt_1),75). orientation(mws1gt_2,162,75). coords(mws1gt_2,37,169,75).  visible(mws1gt_2,75). happensAt(walking(mws1gt_2),75).
orientation(mws1gt_1,160,76). coords(mws1gt_1,79,187,76).  visible(mws1gt_1,76). happensAt(walking(mws1gt_1),76). orientation(mws1gt_2,162,76). coords(mws1gt_2,37,170,76).  visible(mws1gt_2,76). happensAt(walking(mws1gt_2),76).
orientation(mws1gt_1,160,77). coords(mws1gt_1,79,186,77).  visible(mws1gt_1,77). happensAt(walking(mws1gt_1),77). orientation(mws1gt_2,162,77). coords(mws1gt_2,37,171,77).  visible(mws1gt_2,77). happensAt(walking(mws1gt_2),77).
orientation(mws1gt_1,160,78). coords(mws1gt_1,80,186,78).  visible(mws1gt_1,78). happensAt(walking(mws1gt_1),78). orientation(mws1gt_2,162,78). coords(mws1gt_2,36,172,78).  visible(mws1gt_2,78). happensAt(walking(mws1gt_2),78).
orientation(mws1gt_1,160,79). coords(mws1gt_1,81,185,79).  visible(mws1gt_1,79). happensAt(walking(mws1gt_1),79). orientation(mws1gt_2,162,79). coords(mws1gt_2,36,172,79).  visible(mws1gt_2,79). happensAt(walking(mws1gt_2),79).
orientation(mws1gt_1,160,80). coords(mws1gt_1,81,184,80).  visible(mws1gt_1,80). happensAt(walking(mws1gt_1),80). orientation(mws1gt_2,162,80). coords(mws1gt_2,36,173,80).  visible(mws1gt_2,80). happensAt(walking(mws1gt_2),80).
orientation(mws1gt_1,160,81). coords(mws1gt_1,81,184,81).  visible(mws1gt_1,81). happensAt(walking(mws1gt_1),81). orientation(mws1gt_2,162,81). coords(mws1gt_2,35,174,81).  visible(mws1gt_2,81). happensAt(walking(mws1gt_2),81).
orientation(mws1gt_1,160,82). coords(mws1gt_1,81,184,82).  visible(mws1gt_1,82). happensAt(walking(mws1gt_1),82). orientation(mws1gt_2,162,82). coords(mws1gt_2,35,175,82).  visible(mws1gt_2,82). happensAt(walking(mws1gt_2),82).
orientation(mws1gt_1,160,83). coords(mws1gt_1,81,184,83).  visible(mws1gt_1,83). happensAt(walking(mws1gt_1),83). orientation(mws1gt_2,162,83). coords(mws1gt_2,35,176,83).  visible(mws1gt_2,83). happensAt(walking(mws1gt_2),83).
orientation(mws1gt_1,160,84). coords(mws1gt_1,81,184,84).  visible(mws1gt_1,84). happensAt(walking(mws1gt_1),84). orientation(mws1gt_2,162,84). coords(mws1gt_2,35,177,84).  visible(mws1gt_2,84). happensAt(walking(mws1gt_2),84).
orientation(mws1gt_1,160,85). coords(mws1gt_1,81,183,85).  visible(mws1gt_1,85). happensAt(walking(mws1gt_1),85). orientation(mws1gt_2,162,85). coords(mws1gt_2,35,178,85).  visible(mws1gt_2,85). happensAt(walking(mws1gt_2),85).
#show holdsAt/2.