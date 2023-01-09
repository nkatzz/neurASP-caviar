
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

	orientation(wk1gt_0,6,51). coords(wk1gt_0,223,216,51).  visible(wk1gt_0,51). happensAt(walking(wk1gt_0),51). orientation(wk1gt_4,0,51). coords(wk1gt_4,27,209,51).  visible(wk1gt_4,51). happensAt(walking(wk1gt_4),51).
orientation(wk1gt_0,5,52). coords(wk1gt_0,223,216,52).  visible(wk1gt_0,52). happensAt(walking(wk1gt_0),52). orientation(wk1gt_4,0,52). coords(wk1gt_4,27,208,52).  visible(wk1gt_4,52). happensAt(walking(wk1gt_4),52).
orientation(wk1gt_0,169,53). coords(wk1gt_0,223,216,53).  visible(wk1gt_0,53). happensAt(walking(wk1gt_0),53). orientation(wk1gt_4,0,53). coords(wk1gt_4,28,208,53).  visible(wk1gt_4,53). happensAt(walking(wk1gt_4),53).
orientation(wk1gt_0,158,54). coords(wk1gt_0,223,215,54).  visible(wk1gt_0,54). happensAt(walking(wk1gt_0),54). orientation(wk1gt_4,0,54). coords(wk1gt_4,28,208,54).  visible(wk1gt_4,54). happensAt(walking(wk1gt_4),54).
orientation(wk1gt_0,155,55). coords(wk1gt_0,223,214,55).  visible(wk1gt_0,55). happensAt(walking(wk1gt_0),55). orientation(wk1gt_4,0,55). coords(wk1gt_4,28,207,55).  visible(wk1gt_4,55). happensAt(walking(wk1gt_4),55).
orientation(wk1gt_0,158,56). coords(wk1gt_0,223,209,56).  visible(wk1gt_0,56). happensAt(walking(wk1gt_0),56). orientation(wk1gt_4,0,56). coords(wk1gt_4,29,207,56).  visible(wk1gt_4,56). happensAt(walking(wk1gt_4),56).
orientation(wk1gt_0,155,57). coords(wk1gt_0,223,204,57).  visible(wk1gt_0,57). happensAt(walking(wk1gt_0),57). orientation(wk1gt_4,0,57). coords(wk1gt_4,30,207,57).  visible(wk1gt_4,57). happensAt(walking(wk1gt_4),57).
orientation(wk1gt_0,155,58). coords(wk1gt_0,223,203,58).  visible(wk1gt_0,58). happensAt(walking(wk1gt_0),58). orientation(wk1gt_4,0,58). coords(wk1gt_4,31,207,58).  visible(wk1gt_4,58). happensAt(walking(wk1gt_4),58).
orientation(wk1gt_0,155,59). coords(wk1gt_0,223,201,59).  visible(wk1gt_0,59). happensAt(walking(wk1gt_0),59). orientation(wk1gt_4,0,59). coords(wk1gt_4,31,206,59).  visible(wk1gt_4,59). happensAt(walking(wk1gt_4),59).
orientation(wk1gt_0,155,60). coords(wk1gt_0,223,199,60).  visible(wk1gt_0,60). happensAt(walking(wk1gt_0),60). orientation(wk1gt_4,0,60). coords(wk1gt_4,31,206,60).  visible(wk1gt_4,60). happensAt(walking(wk1gt_4),60).
orientation(wk1gt_0,148,61). coords(wk1gt_0,223,197,61).  visible(wk1gt_0,61). happensAt(walking(wk1gt_0),61). orientation(wk1gt_4,0,61). coords(wk1gt_4,35,205,61).  visible(wk1gt_4,61). happensAt(walking(wk1gt_4),61).
orientation(wk1gt_0,148,62). coords(wk1gt_0,223,194,62).  visible(wk1gt_0,62). happensAt(walking(wk1gt_0),62). orientation(wk1gt_4,0,62). coords(wk1gt_4,35,204,62).  visible(wk1gt_4,62). happensAt(walking(wk1gt_4),62).
orientation(wk1gt_0,143,63). coords(wk1gt_0,223,192,63).  visible(wk1gt_0,63). happensAt(walking(wk1gt_0),63). orientation(wk1gt_4,0,63). coords(wk1gt_4,35,204,63).  visible(wk1gt_4,63). happensAt(walking(wk1gt_4),63).
orientation(wk1gt_0,143,64). coords(wk1gt_0,223,189,64).  visible(wk1gt_0,64). happensAt(walking(wk1gt_0),64). orientation(wk1gt_4,0,64). coords(wk1gt_4,35,204,64).  visible(wk1gt_4,64). happensAt(walking(wk1gt_4),64).
orientation(wk1gt_0,143,65). coords(wk1gt_0,222,186,65).  visible(wk1gt_0,65). happensAt(walking(wk1gt_0),65). orientation(wk1gt_4,0,65). coords(wk1gt_4,35,203,65).  visible(wk1gt_4,65). happensAt(walking(wk1gt_4),65).
orientation(wk1gt_0,140,66). coords(wk1gt_0,222,183,66).  visible(wk1gt_0,66). happensAt(walking(wk1gt_0),66). orientation(wk1gt_4,0,66). coords(wk1gt_4,35,203,66).  visible(wk1gt_4,66). happensAt(walking(wk1gt_4),66).
orientation(wk1gt_0,140,67). coords(wk1gt_0,222,181,67).  visible(wk1gt_0,67). happensAt(walking(wk1gt_0),67). orientation(wk1gt_4,0,67). coords(wk1gt_4,36,203,67).  visible(wk1gt_4,67). happensAt(walking(wk1gt_4),67).
orientation(wk1gt_0,138,68). coords(wk1gt_0,221,179,68).  visible(wk1gt_0,68). happensAt(walking(wk1gt_0),68). orientation(wk1gt_4,0,68). coords(wk1gt_4,36,202,68).  visible(wk1gt_4,68). happensAt(walking(wk1gt_4),68).
orientation(wk1gt_0,136,69). coords(wk1gt_0,220,177,69).  visible(wk1gt_0,69). happensAt(walking(wk1gt_0),69). orientation(wk1gt_4,0,69). coords(wk1gt_4,36,201,69).  visible(wk1gt_4,69). happensAt(walking(wk1gt_4),69).
orientation(wk1gt_0,136,70). coords(wk1gt_0,220,175,70).  visible(wk1gt_0,70). happensAt(walking(wk1gt_0),70). orientation(wk1gt_4,0,70). coords(wk1gt_4,36,200,70).  visible(wk1gt_4,70). happensAt(walking(wk1gt_4),70).
orientation(wk1gt_0,136,71). coords(wk1gt_0,219,173,71).  visible(wk1gt_0,71). happensAt(walking(wk1gt_0),71). orientation(wk1gt_4,0,71). coords(wk1gt_4,36,199,71).  visible(wk1gt_4,71). happensAt(walking(wk1gt_4),71).
orientation(wk1gt_0,136,72). coords(wk1gt_0,219,170,72).  visible(wk1gt_0,72). happensAt(walking(wk1gt_0),72). orientation(wk1gt_4,0,72). coords(wk1gt_4,36,199,72).  visible(wk1gt_4,72). happensAt(walking(wk1gt_4),72).
orientation(wk1gt_0,136,73). coords(wk1gt_0,219,168,73).  visible(wk1gt_0,73). happensAt(walking(wk1gt_0),73). orientation(wk1gt_4,0,73). coords(wk1gt_4,36,199,73).  visible(wk1gt_4,73). happensAt(walking(wk1gt_4),73).
orientation(wk1gt_0,136,74). coords(wk1gt_0,218,166,74).  visible(wk1gt_0,74). happensAt(walking(wk1gt_0),74). orientation(wk1gt_4,0,74). coords(wk1gt_4,36,199,74).  visible(wk1gt_4,74). happensAt(walking(wk1gt_4),74).
orientation(wk1gt_0,136,75). coords(wk1gt_0,218,162,75).  visible(wk1gt_0,75). happensAt(walking(wk1gt_0),75). orientation(wk1gt_4,0,75). coords(wk1gt_4,36,198,75).  visible(wk1gt_4,75). happensAt(walking(wk1gt_4),75).
#show holdsAt/2.